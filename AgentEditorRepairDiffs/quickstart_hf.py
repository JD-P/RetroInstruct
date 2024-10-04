import ast
import random
import json
import datasets
from render_block import render_block

def is_list_of_tuples(s):
    try:
        # Parse the string into an AST node
        node = ast.parse(s, mode='eval')

        # Check if the node is a list
        if not isinstance(node.body, ast.List):
            return False

        # Check if all elements in the list are tuples with the correct structure
        for element in node.body.elts:
            if not isinstance(element, ast.Tuple):
                return False
            if len(element.elts) != 3:
                return False
            if not isinstance(element.elts[0], ast.Constant) or not isinstance(element.elts[0].value, int):
                return False
            if not isinstance(element.elts[1], ast.Constant) or not isinstance(element.elts[1].value, int):
                return False
            if not isinstance(element.elts[2], ast.Constant) or not isinstance(element.elts[2].value, str):
                return False

        return True
    except SyntaxError:
        return False

def is_list_of_strings(s):
    try:
        # Parse the string into an AST node
        node = ast.parse(s, mode='eval')

        # Check if the node is a list
        if not isinstance(node.body, ast.List):
            return False

        # Check if all elements in the list are strings
        for element in node.body.elts:
            if not isinstance(element, ast.Str):
                return False

        return True
    except SyntaxError:
        return False

def split_lines_with_custom_newlines(content, newline_chars):
    """
    Splits the lines in a file based on a custom set of newline characters.

    :param file_path: Path to the file to be read.
    :param newline_chars: A set of characters to be considered as newlines.
    :return: A list of lines split based on the custom newline characters.
    """
    lines = []
    current_line = []

    for char in content:
        current_line.append(char)
        if char in newline_chars:
            lines.append(''.join(current_line))
            current_line = []
    # Add the last line if there is any remaining content
    if current_line:
        lines.append(''.join(current_line))

    return lines

def render(file_content, filepath):
    """Render the text block based on the current line number position and the window size."""
    total_lines = len(file_content)
    start_line = 0
    end_line = total_lines

    # Create the rendered text block
    rendered_text = f"#File: {filepath} ({total_lines} lines total)#\n"
    rendered_text += f"'''({start_line} lines above)\n"
    for i in range(start_line, end_line):
        rendered_text += f"{i+1} {file_content[i]}"
    rendered_text += f"\n({total_lines - end_line} lines below)\n'''"
    return rendered_text

data = datasets.load_dataset("jdpressman/retro-weave-agent-editor-repair-diffs-v0.1")["train"]

for row in data:
    texts = [
        (row["text_corrupted"], row["text_corrupted_filename"]),
        (row["text_herring1"], row["text_herring1_filename"]),
        (row["text_herring2"], row["text_herring2_filename"])
    ]
    random.shuffle(texts)
    blocks = []
    for text, filepath in texts:
        file_content = split_lines_with_custom_newlines(text, "\n\r\x0b\x0c")
        view_title = f"WeaveEditor ({filepath})"
        content = render(file_content, filepath)
        event_block = {"type":"observation",
                       "title": view_title,
                       "content":content}
        observation = render_block(event_block)
        blocks.append(observation)
    agenda = row['orientation_agenda'].format(filename=row["text_corrupted_filename"])
    diagnosis = ''.join([f"- {op}" for op in row["operations"].splitlines(True)])
    program = f"\"\"\"\n{agenda}\n\n{diagnosis}\"\"\""
    tick_number = random.randint(8,200)
    block_index = random.choice([8,9,10]) * tick_number
    orientation_block = {"type":"orientation",
                         "timestamp":row["orientation_timestamp"],
                         "metadata":{"tick_number":tick_number,
                                     "block_index":block_index,
                                     "working_directory":"/app/"},
                         "program":program}
    orientation = render_block(orientation_block)
    blocks.append(orientation)

    edit_method = random.choice(["diff", "diff", "edit()"])
    program = f"def {row['action_callback_name']}(agent):\n"
    if edit_method == "edit()":
        docstring = row["action_edit_docstring"].format(
            filename=row["text_corrupted_filename"]
        )
        program += f'    """{docstring}"""\n'
        program += f"    editor = agent.tools['editor-{row['text_corrupted_filename']}']\n"
        if is_list_of_tuples(row["edits"]):
            edits = eval(row["edits"])
        else:
            raise ValueError("Expected list of edits but got something else. Attempted ACE?")
        for edit in edits:
            program += f"    editor.edit({edit[0]}, {edit[1]}, {repr(edit[2])})\n"
        program += "\n"
    else:
        docstring = row["action_unidiff_docstring"].format(
            filename=row["text_corrupted_filename"]
        )
        program += f'    """{docstring}"""\n'
        program += f"    editor = agent.tools['editor-{row['text_corrupted_filename']}']\n"
        program += "    diff_lines = [\n"
        if is_list_of_strings(row["diff"]):
            diff = eval(row["diff"])
        else:
            raise ValueError("Expected list of strings but got something else. Attempted ACE?")
        for line in diff:
            program += f"        {repr(line)}\n"
        program += "    ]\n"
        program += "    editor.unidiff_edit(diff_lines)\n"
    callback_desc = row['action_callback_description'].format(
        filename=row['text_corrupted_filename']
    )
    program += f"agent.add_action({repr(callback_desc)}, {row['action_callback_name']})"
    action_block = {"type":"action",
                    "timestamp":row["action_timestamp"],
                    "program":program}
    action = render_block(action_block)
    blocks.append(action)

    
    file_content = split_lines_with_custom_newlines(row["text_clean"], "\n\r\x0b\x0c")
    view_title = f"WeaveEditor ({row['text_corrupted_filename']})"
    content = render(file_content, filepath)
    observation_block = {"type":"observation",
                         "title":view_title,
                         "content":content}
    observation = render_block(observation_block)
    blocks.append(observation)
    
    for block in blocks:
        print(block, end="")
