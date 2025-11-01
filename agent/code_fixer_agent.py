import re
from .react_agent import generate_text, run_sandboxed
import textwrap

def _parse_model_code_output(resp: str) -> str: #private function
    blocks = re.findall(r"```(?:python)?(.*?)```", resp, re.DOTALL)
    if blocks:
        final_code = blocks[-1].strip() #we select the last code-block
    else:
        final_code = resp.strip()
    final_code = textwrap.dedent(final_code).strip()
    return final_code

def _test_code(code: str) -> str:
    doctest_runner = (
            "\n\n"
            + "if __name__ == '__main__':\n"
            + "    import doctest, sys\n"
            + "    res = doctest.testmod()\n"
            + "    if res.failed == 0:\n"
            + "        print('DOCTEST_OK')\n"
            + "    else:\n"
            + "        print('DOCTEST_FAILED', res)\n"
    )

    wrapper = (code + doctest_runner)
    return wrapper

def invoke(inputs: dict) -> dict:
    buggy_code = inputs.get("buggy_code", "")
    if not buggy_code:
        return {"fixed_code": ""}

    base_prompt = f"""You are an expert Python developer and code fixer.
Your job: fix the given Python code so that it executes without SyntaxError and (if present) passes doctests in the file.
IMPORTANT:
- Return ONLY the full corrected Python file/code. Do NOT return only the function body.
- Do NOT add surrounding markdown or explanations.
- Ensure indentation is correct and the code is self-contained (include imports if necessary).

Buggy code:
---
{buggy_code}
---
When executed, it produced an error or unexpected output. Return the corrected Python code only.
"""

    max_attempts = 3
    last_candidate = ""
    for attempt in range(max_attempts):
        response = generate_text(base_prompt, max_new_tokens=512, temperature=0.01)
        final_code = _parse_model_code_output(response)
        last_candidate = final_code

        # test the code with doctest
        wrapper = _test_code(final_code)
        result = run_sandboxed(wrapper, timeout=8)

        stdout = result.get("stdout", "")
        stderr = result.get("stderr", "")
        return_code = result.get("returncode", 1)

        if return_code == 0:
            # Accept if all doctests passed or if there are no doctests at all
            return {"fixed_code": final_code}

        # If the code fails, provide the model with feedback (error details) and ask it to fix the code again (agent have two more attempts max)
        feedback = f"Execution returncode={return_code}\nstdout:\n{stdout}\nstderr:\n{stderr}"
        base_prompt += (
                f"\nPrevious attempt:\n---\n{final_code}\n---\n"
                f"When executed it returned:\n{feedback}\n"
                "Please provide a corrected version (only runnable Python code)."
        )
    return {"fixed_code": last_candidate}
