import unittest

from seminar_demo.safe_execution import execute_safely, strip_code_fences


class SafeExecutionTests(unittest.TestCase):
    def test_strips_code_fences(self):
        self.assertEqual(strip_code_fences("```python\nresult = 2 + 3\n```"), "result = 2 + 3")

    def test_safe_arithmetic(self):
        execution = execute_safely("result = round(math.sqrt(81) + 1.25, 2)")
        self.assertEqual(execution.status, "success")
        self.assertEqual(execution.result, 10.25)

    def test_safe_rule_conditionals(self):
        code = """
score = 0
age = 76
if age >= 75:
    score += 2
result = score
"""
        execution = execute_safely(code)
        self.assertEqual(execution.status, "success")
        self.assertEqual(execution.result, 2)

    def test_rejects_import_and_filesystem(self):
        execution = execute_safely("import os\nresult = os.listdir('.')")
        self.assertEqual(execution.status, "rejected")
        self.assertIn("Forbidden syntax", execution.error)

    def test_rejects_dynamic_builtin_access(self):
        execution = execute_safely("result = __import__('os').system('id')")
        self.assertEqual(execution.status, "rejected")

    def test_requires_result_assignment(self):
        execution = execute_safely("value = 2 + 2")
        self.assertEqual(execution.status, "rejected")
        self.assertIn("must assign", execution.error)


if __name__ == "__main__":
    unittest.main()
