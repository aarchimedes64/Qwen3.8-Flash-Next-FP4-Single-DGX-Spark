import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RecipeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="spark recipe ")
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name) / "checkout with spaces"
        shutil.copytree(ROOT, self.repo, ignore=shutil.ignore_patterns(".git", ".env", "__pycache__"))
        self.bin = Path(self.tmp.name) / "bin"
        self.bin.mkdir()
        self.env = dict(os.environ, HF_TOKEN="", API_KEY="")
        for key in ("IMAGE", "BIND", "PORT", "MAX_MODEL_LEN", "MAX_NUM_SEQS",
                    "MAX_NUM_BATCHED_TOKENS", "GPU_MEMORY_UTILIZATION", "HF_CACHE",
                    "VLLM_CACHE", "FLASHINFER_CACHE", "TRITON_CACHE", "B12X_CACHE"):
            self.env.pop(key, None)
        self.env["PATH"] = str(self.bin) + os.pathsep + self.env["PATH"]
        self.capture = Path(self.tmp.name) / "capture.jsonl"
        self.env["CAPTURE"] = str(self.capture)

    def executable(self, name, body):
        path = self.bin / name
        path.write_text("#!" + sys.executable + "\n" + body)
        path.chmod(0o755)

    def recorder(self, name):
        self.executable(name, "import json, os, sys\n"
                        "with open(os.environ['CAPTURE'], 'a') as f: f.write(json.dumps(sys.argv[1:]) + '\\n')\n")

    def run_script(self, script, *args):
        return subprocess.run(["bash", str(self.repo / script), *args], env=self.env,
                              cwd=self.tmp.name, text=True, capture_output=True, check=True)

    def calls(self):
        return [json.loads(line) for line in self.capture.read_text().splitlines()]

    def config(self):
        return json.loads(subprocess.check_output(
            ["docker", "compose", "--project-directory", str(self.repo), "--env-file",
             str(self.repo / ".env.sample"), "-f", str(self.repo / "compose.yaml"),
             "config", "--format", "json"], env=self.env, text=True))

    def test_nvidia_command_parity(self):
        self.recorder("vllm")
        self.run_script("scripts/serve.sh")
        args = self.calls()[0]
        self.assertEqual(args[:2], ["serve", "nvidia/Qwen3.8-Flash-Next-FP4-Local"])
        # NVIDIA model-card reference with the recipe's port 8888 default.
        expected = {
            "--host": "0.0.0.0", "--port": "8888", "--tensor-parallel-size": "1",
            "--pipeline-parallel-size": "1", "--mamba-cache-mode": "align",
            "--dtype": "bfloat16", "--kv-cache-dtype": "fp8",
            "--quantization": "modelopt_mixed", "--block-size": "16",
            "--load-format": "b12x", "--max-model-len": "262144",
            "--max-num-seqs": "8", "--max-num-batched-tokens": "4096",
            "--speculative-config": '{"method":"mtp","num_speculative_tokens":4}',
            "--gdn-decode-kernel": "b12x", "--linear-backend": "b12x",
            "--moe-backend": "b12x", "--mm-encoder-tp-mode": "data",
            "--reasoning-parser": "qwen3", "--tool-call-parser": "qwen3_xml",
            "--compilation-config": '{"pass_config":{"fuse_act_quant":true}}',
            "--gpu-memory-utilization": "0.8",
        }
        flags = {"--enable-prefix-caching", "--enable-chunked-prefill",
                 "--no-enable-flashinfer-autotune", "--enable-auto-tool-choice"}
        actual = {}
        pos = 2
        while pos < len(args):
            key = args[pos]
            self.assertNotIn(key, actual)
            if key in flags:
                actual[key] = None
                pos += 1
            else:
                actual[key] = args[pos + 1]
                pos += 2
        self.assertEqual(actual, dict(expected, **{flag: None for flag in flags}))

    def test_serving_overrides_and_key_with_spaces(self):
        self.recorder("vllm")
        self.env.update(API_KEY="test key $literal", PORT="9000", MAX_NUM_SEQS="2")
        self.run_script("scripts/serve.sh")
        args = self.calls()[0]
        self.assertEqual(args[args.index("--port") + 1], "9000")
        self.assertEqual(args[args.index("--max-num-seqs") + 1], "2")
        self.assertEqual(args[-2:], ["--api-key", "test key $literal"])

    def test_compose_nvidia_environment_and_persistence(self):
        service = self.config()["services"]["vllm"]
        env = service["environment"]
        expected = {
            "VLLM_PLE_TABLE_MEMORY": "disk", "B12X_DISK_BACKEND": "io_uring",
            "CUTE_DSL_ARCH": "sm_121a", "SAFETENSORS_FAST_GPU": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "VLLM_SSM_CONV_STATE_LAYOUT": "DS",
            "VLLM_USE_AOT_COMPILE": "1", "VLLM_USE_MEGA_AOT_ARTIFACT": "1",
            "VLLM_USE_V2_MODEL_RUNNER": "1", "B12X_POLICY_MODE": "auto",
            "VLLM_MXFP8_LM_HEAD": "1", "VLLM_B12X_MOE_FP4_LAYER_MAX_INPUT_SCALE": "w13",
            "VLLM_B12X_MOE_FP4_FORCE_A16": "1", "B12X_W4A16_A4_PREFILL": "1",
        }
        self.assertEqual({key: env[key] for key in expected}, expected)
        self.assertEqual(service["image"], "eugr/spark-vllm-b12x:latest")
        self.assertEqual(service["restart"], "unless-stopped")
        self.assertEqual(service["network_mode"], "host")
        self.assertEqual(service["ipc"], "host")
        self.assertTrue(service["init"])
        self.assertEqual(service["gpus"][0]["count"], -1)
        self.assertEqual({v["target"] for v in service["volumes"]}, {
            "/root/.cache/huggingface", "/root/.cache/vllm", "/root/.cache/flashinfer",
            "/root/.triton", "/root/.cache/b12x", "/opt/recipe/serve.sh"})
        script = next(v for v in service["volumes"] if v["target"] == "/opt/recipe/serve.sh")
        self.assertTrue(script["read_only"])

    def test_compose_shell_overrides(self):
        self.env.update(PORT="9001", API_KEY="test-key", HF_CACHE="/tmp/custom-cache")
        service = self.config()["services"]["vllm"]
        self.assertEqual(service["environment"]["PORT"], "9001")
        self.assertEqual(service["environment"]["API_KEY"], "test-key")
        volume = next(v for v in service["volumes"] if v["target"] == "/root/.cache/huggingface")
        self.assertEqual(volume["source"], "/tmp/custom-cache")

    def test_dry_run_never_launches_or_prints_tokens(self):
        self.recorder("docker")
        self.env.update(HF_TOKEN="hf_test_secret", API_KEY="test_secret")
        result = self.run_script("start.sh", "--no-launch")
        self.assertNotIn("hf_test_secret", result.stdout + result.stderr)
        self.assertNotIn("test_secret", result.stdout + result.stderr)
        self.assertEqual(len(self.calls()), 2)
        self.assertEqual(self.calls()[-1][-2:], ["config", "--quiet"])
        self.assertEqual((self.repo / ".env").stat().st_mode & 0o777, 0o600)

    def test_detached_start_and_explicit_pull(self):
        self.recorder("docker")
        self.executable("uname", "import sys\nprint('Linux' if sys.argv[1] == '-s' else 'aarch64')\n")
        self.run_script("start.sh")
        self.assertEqual(self.calls()[-1][-3:], ["up", "--detach", "vllm"])
        self.assertFalse(any("pull" in args for args in self.calls()))
        self.capture.unlink()
        self.run_script("start.sh", "--pull")
        self.assertEqual(self.calls()[-2][-2:], ["pull", "vllm"])
        self.assertEqual(self.calls()[-1][-3:], ["up", "--detach", "vllm"])

    def test_lifecycle_helpers(self):
        self.recorder("docker")
        (self.repo / ".env").write_text("")
        for script, tail in [("stop.sh", ["stop", "vllm"]),
                             ("logs.sh", ["logs", "--follow", "--tail", "100", "vllm"]),
                             ("status.sh", ["ps", "--all"])]:
            self.run_script(script)
            self.assertEqual(self.calls()[-1][-len(tail):], tail)


if __name__ == "__main__":
    unittest.main()
