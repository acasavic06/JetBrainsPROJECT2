from transformers import AutoTokenizer, AutoModelForCausalLM
import subprocess
import tempfile
import os
import textwrap

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True, device_map="auto")

def run_sandboxed(code: str, timeout: int = 5) -> dict:
    """
        with this function we will return dictionary with three elements:
        {
          "returncode": int,
          "stdout": str,
          "stderr": str
        }
    """
    code = textwrap.dedent(code).strip() + "\n"
    temporaryFile = None
    try:
        temporaryFile = tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="w", encoding="utf-8")
        temporaryFile.write(code)
        temporaryFile.flush()
        temporaryFile.close()
        result = subprocess.run(
            ["python", temporaryFile.name],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except Exception as e:
        return {"returncode": 1, "stdout": "", "stderr": f"Exception: {e}"}
    finally:
        if temporaryFile is not None:
            try:
                os.unlink(temporaryFile.name) #to delete temporary file
            except Exception:
                pass

def generate_text(prompt: str, max_new_tokens: int = 512, temperature: float = 0.01) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=max(temperature,0.01),
        do_sample=False #always select the most likely output
    )
    resp = tokenizer.decode(outputs[0], skip_special_tokens=True) #decode model output to string
    return resp
