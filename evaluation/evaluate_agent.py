from datasets import load_dataset
from agent.code_fixer_agent import invoke
from agent.react_agent import run_sandboxed
import warnings
warnings.filterwarnings("ignore")

def evaluate_agent():
    tests= load_dataset("bigcode/humanevalpack", split="test[:50]")
    passed = 0
    failed_tests=set()
    total = len(tests)

    for i, test in enumerate(tests):
        buggy = test["prompt"]
        print(f"\n--- Example {i+1}/{total} ---")
        print(buggy.strip()[:300])

        res = invoke({"buggy_code": buggy})
        fixed = res.get("fixed_code", "")
        if not fixed:
            failed_tests.add(i)
            print("Agent returned empty fix.")
            continue

        result = run_sandboxed(fixed, timeout=8)
        return_code = result["returncode"]
        stderr = result["stderr"]

        success = (return_code == 0)
        print("Result:", "PASS" if success else "FAIL")
        if not success:
            failed_tests.add(i)
            print(stderr[:400])
        else:
            passed += 1

    print(f"\nFinal score: {passed}/{total} = {passed/total:.2%}")
    if failed_tests:
        print(f"\nFailed tests: {sorted(failed_tests)}")
    else:
        print("\nAll tests passed!")

if __name__ == "__main__":
    evaluate_agent()
