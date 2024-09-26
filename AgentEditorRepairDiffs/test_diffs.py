import os
import re
import unittest
import difflib
from editor import WeaveEditor, parse_diff, parse_diff_header

def generate_diff(original_lines, modified_lines):
    diff = difflib.unified_diff(original_lines, modified_lines, lineterm='\n')
    return list(diff)

class TestWeaveEditorDiff(unittest.TestCase):
    def setUp(self):
        self.agent = type('Agent', (object,), {'tools': [], 'add_observation_view': lambda self, x, y: None, 'remove_observation_view': lambda self, x: None})()
        with open("test_file.txt", "w") as outfile:
            outfile.write("Line 1\nLine 2\nLine 3\n")
        self.editor = WeaveEditor(self.agent, 'test_file.txt')

    def tearDown(self):
        self.editor.close()
        os.remove("test_file.txt")

    def test_generate_diff(self):
        original_lines = ["Line 1\n", "Line 2\n", "Line 3\n"]
        modified_lines = ["Line 1\n", "Modified Line 2\n", "Line 3\n"]
        diff_lines = generate_diff(original_lines, modified_lines)
        expected_diff = [
            "--- \n",
            "+++ \n",
            "@@ -1,3 +1,3 @@\n",
            " Line 1\n",
            "-Line 2\n",
            "+Modified Line 2\n",
            " Line 3\n"
        ]
        self.assertEqual(diff_lines, expected_diff)

    def test_parse_diff_header(self):
        line = "@@ -1,3 +1,3 @@\n"
        result = parse_diff_header(line)
        expected_result = (1, 3, 1, 3)
        self.assertEqual(result, expected_result)

        line = "@@ -1 +1 @@\n"
        result = parse_diff_header(line)
        expected_result = (1, 1, 1, 1)
        self.assertEqual(result, expected_result)

        line = "@@ -1,3 +1 @@\n"
        result = parse_diff_header(line)
        expected_result = (1, 3, 1, 1)
        self.assertEqual(result, expected_result)

        line = "@@ -1 +1,3 @@\n"
        result = parse_diff_header(line)
        expected_result = (1, 1, 1, 3)
        self.assertEqual(result, expected_result)

    def test_parse_diff(self):
        diff_lines = [
            "--- \n",
            "+++ \n",
            "@@ -1,3 +1,3 @@\n",
            " Line 1\n",
            "-Line 2\n",
            "+Modified Line 2\n",
            " Line 3\n"
        ]
        edits = parse_diff(diff_lines)
        expected_edits = [(1, 3, 'Line 1\nModified Line 2\nLine 3\n'),]
        self.assertEqual(edits, expected_edits)

    def test_apply_edits(self):
        original_lines = self.editor.file_content
        modified_lines = ["Line 1\n", "Modified Line 2\n", "Line 3\n"]
        diff_lines = generate_diff(original_lines, modified_lines)
        self.editor.unidiff_edit(diff_lines)
        expected_content = ["Line 1\n", "Modified Line 2\n", "Line 3\n"]
        self.assertEqual(self.editor.file_content, expected_content)

if __name__ == '__main__':
    unittest.main()
