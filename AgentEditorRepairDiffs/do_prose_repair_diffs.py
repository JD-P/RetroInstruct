import random
import string
import json
import re
import os
import time
import difflib
import tempfile
import git
from git import Repo
from datetime import datetime
import subprocess
from filename_builder import FileNameBuilder
from editor import WeaveEditor, parse_diff
from transformers import AutoTokenizer
from tqdm import tqdm

# Load the dictionary file
with open('/usr/share/dict/words') as f:
    words = f.read().splitlines()

# Filter out words with apostrophes or similar
valid_words = [word.lower() for word in words if not any(c in string.punctuation for c in word)]

def transpose_substrings(passage1, passage2):
    # Choose random substrings from both passages
    substring1_start = random.randint(0, len(passage1) - 1)
    substring1_end = random.randint(substring1_start, substring1_start + 512)
    substring1 = passage1[substring1_start:substring1_end]

    substring2_start = random.randint(0, len(passage2) - 1)
    substring2_end = random.randint(substring2_start, substring2_start + 512)
    substring2 = passage2[substring2_start:substring2_end]

    # Replace the substring in the first passage with the substring from the second passage
    corrupted_passage = passage1.replace(substring1, substring2, 1)

    return corrupted_passage, (substring1_start, substring1_end, len(substring2))


def swap_substrings(passage):
    # Find two random substrings in the passage
    substring1_start = random.randint(0, len(passage) - 1)
    substring1_end = random.randint(substring1_start + 1, substring1_start + 512)
    substring1 = passage[substring1_start:substring1_end]

    substring2_start = random.randint(0, len(passage) - 1)
    substring2_end = random.randint(substring2_start + 1, substring2_start + 512)
    substring2 = passage[substring2_start:substring2_end]

    # If the two substrings are the same, return the original passage
    if substring1 == substring2:
        return passage, tuple()
    
    # Use a temporary string to avoid replacing the wrong occurrence of substring2
    temp_string = 'TEMP_STRING' + str(random.randint(0, 9))

    # Replace the first occurrence of each substring with the other substring
    corrupted_passage = passage.replace(substring1, temp_string, 1)
    corrupted_passage = corrupted_passage.replace(substring2, substring1, 1)
    corrupted_passage = corrupted_passage.replace(temp_string, substring2, 1)

    return corrupted_passage, (substring1_start, substring1_end, substring2_start, substring2_end)


def substring2gibberish(passage):
    # Choose a random substring from the passage
    substring_start = random.randint(0, len(passage) - 1)
    substring_end = random.randint(substring_start, substring_start + 64)
    substring = passage[substring_start:substring_end]

    # Generate a gibberish ASCII string of the same length as the substring
    gibberish = ''.join(random.choice(string.printable) for _ in range(len(substring)))

    # Replace all instances of the substring in the passage with the gibberish string
    corrupted_passage = passage.replace(substring, gibberish)

    return corrupted_passage, (substring_start, substring_end)


def adjacent_substring_swap(passage):
    # Choose a random substring from the passage
    substring_start = random.randint(0, len(passage) - 1)
    substring_end = random.randint(substring_start, substring_start + 256)
    substring = passage[substring_start:substring_end]

    # If the substring is empty, return the original passage
    if not substring:
        return passage, tuple()

    # Split the substring in half and swap the halves
    midpoint = len(substring) // 2
    first_half = substring[:midpoint]
    second_half = substring[midpoint:]
    swapped_substring = second_half + first_half

    # Replace the first occurrence of the substring in the passage with the swapped substring
    corrupted_passage = passage.replace(substring, swapped_substring, 1)

    return corrupted_passage, (substring_start, substring_end)


def delete_substring(passage):
    # Choose a random substring from the passage
    substring_start = random.randint(0, len(passage) - 1)
    substring_end = random.randint(substring_start, substring_start + 64)
    substring = passage[substring_start:substring_end]

    # Replace the first occurrence of the substring in the passage with an empty string
    corrupted_passage = passage.replace(substring, '', 1)

    return corrupted_passage, (substring_start, substring_end)


def insert_spurious_html_xml_tag(passage):
    # Choose a random word from the dictionary to use as the tag name
    tag_name = random.choice(valid_words)

    # Choose a random position in the passage to insert the tag
    insert_index = random.randint(0, len(passage))

    # Generate a random tag type (opening, closing, or self-closing)
    tag_types = ['<{}>'.format(tag_name), '</{}>'.format(tag_name), '<{}/>'.format(tag_name)]
    tag = random.choice(tag_types)

    # Insert the tag into the passage
    corrupted_passage = passage[:insert_index] + tag + passage[insert_index:]

    return corrupted_passage, (insert_index, tag)


def delete_whitespace_character(passage):
    # Find all whitespace characters in the passage
    whitespace_chars = [c for c in enumerate(passage) if c[1].isspace()]

    # If there are no whitespace characters, return the original passage
    if not whitespace_chars:
        return passage, tuple()

    # Choose a random whitespace character to delete
    split_index, whitespace_char = random.choice(whitespace_chars)

    corrupted_passage = passage[:split_index] + passage[split_index + 1:]
    
    return corrupted_passage, (split_index,)


def duplicate_word(passage):
    # Split the passage into a list of words
    words = passage.split(" ")

    # If there are no words, return the original passage
    if not words:
        return passage, tuple()

    # Choose a random word to duplicate
    word_index = random.randint(0, len(words) - 1)
    word = words[word_index]

    # Insert the duplicate word at a random position
    insert_index = random.randint(0, len(words))
    words.insert(insert_index, word)

    # Join the words back into a single string
    corrupted_passage = ' '.join(words)

    return corrupted_passage, (word_index,)


def insert_printable_ascii_character(passage):
    # Choose a random position in the passage to insert a symbol
    insert_index = random.randint(0, len(passage))

    # Choose a random printable ASCII character to insert
    symbol = random.choice(string.printable)

    # Insert the symbol into the passage
    corrupted_passage = passage[:insert_index] + symbol + passage[insert_index:]

    return corrupted_passage, (insert_index,)


def shuffle_word_middle(passage):
    # Split the passage into a list of words
    words = passage.split(" ")

    # If there are no words, return the original passage
    if not words:
        return passage, tuple()

    # Choose a random word to shuffle
    word_index = random.randint(0, len(words) - 1)
    word = words[word_index]

    # Shuffle the middle characters of the word
    if len(word) > 2:
        mid_chars = list(word[1:-1])
        random.shuffle(mid_chars)
        shuffled_word = word[0] + ''.join(mid_chars) + word[-1]
    else:
        shuffled_word = word

    # Replace all instances of the original word with the shuffled version
    corrupted_passage = ' '.join([shuffled_word if w == word else w for w in words])

    return corrupted_passage, (word_index,)


def swap_capitalization(passage):
    # Choose a random index in the passage to swap the capitalization of a letter
    index = random.randint(0, len(passage) - 1)

    # If the character at the chosen index is a letter, swap its capitalization
    if passage[index].isalpha():
        if passage[index].islower():
            corrupted_passage = passage[:index] + passage[index].upper() + passage[index+1:]
        else:
            corrupted_passage = passage[:index] + passage[index].lower() + passage[index+1:]
    else:
        corrupted_passage = passage

    return corrupted_passage, (index,)


def insert_punctuation(passage):
    # Choose a random index in the passage to insert punctuation
    index = random.randint(0, len(passage))

    # Choose a random punctuation character to insert
    punctuation = random.choice(string.punctuation)

    # Insert the punctuation character at the chosen index
    corrupted_passage = passage[:index] + punctuation + passage[index:]

    return corrupted_passage, (index,)


def adjacent_word_swap(passage):
    # Split the passage into a list of words
    words = passage.split(" ")

    # If there is only one word, return the original passage
    if len(words) <= 1:
        return passage, tuple()

    # Choose a random index in the list of words to swap with the adjacent word
    index = random.randint(0, len(words) - 2)

    # Swap the chosen word with the adjacent word
    words[index], words[index + 1] = words[index + 1], words[index]

    # Join the list of words back into a single string
    corrupted_passage = ' '.join(words)

    return corrupted_passage, (index,)


def random_number_replacement(passage):
    # Find all numbers in the passage using a regex
    numbers = re.findall(r'\d+', passage)

    # If there are no numbers in the passage, return the original passage
    if not numbers:
        return passage, tuple()

    # Choose a random number to replace
    number = random.choice(numbers)

    # Replace the chosen number with a random other number
    new_number = str(random.randint(0, int(number) * 10))
    corrupted_passage = passage.replace(number, new_number, 1)

    return corrupted_passage, (new_number,)


corruption_passes = [
    transpose_substrings,
    substring2gibberish,
    adjacent_substring_swap,
    delete_substring,
    insert_spurious_html_xml_tag,
    delete_whitespace_character,
    duplicate_word,
    insert_printable_ascii_character,
    shuffle_word_middle,
    swap_capitalization,
    insert_punctuation,
    adjacent_word_swap,
    random_number_replacement,
    swap_substrings
]


def do_corruption(passage, passages, directions_trace=False):
    corruption_function = random.choice(corruption_passes)
    
    if corruption_function == transpose_substrings:
        directions = [
            "Undo substring transpose at [{},{}] with replacement by {} character string.",
            "The span originally at [{}, {}] was replaced by a {} character span.",
            "Span replacement at [{}, {}] resulting in substring of length {}.",
            "Near [{}, {}] a string transposition between two passages inserted {} letters.",
            "A string from another passage was substituted in for a string from this passage.",
            "A substring from one passage was transposed onto the current passage.",
            "Substring transposition took place.",
            "Reverse a substring transposition."
        ]
        passage2 = random.choice(passages)["text"]
        corrupted, direction_params = corruption_function(passage, passage2)
        direction = random.choice(directions)
        if "{}" in direction:
            direction = direction.format(*direction_params)
        if directions_trace:
            direction = "transpose_substrings: " + direction
    elif corruption_function == substring2gibberish:
        directions = [
            "Replace substring gibberish at [{},{}] with inferred original text.",
            "Span near ({},{}) has been replaced with noise.",
            "Undo corruption near ({}, {}).",
            "{},{} has damaged spans.",
            "Reverse gibberish span.",
            "A span was turned into gibberish.",
            "Noise span detected.",
            "Obvious corruption found in passage, preparing infill..."
        ]
        corrupted, direction_params = corruption_function(passage)
        direction = random.choice(directions)
        if "{}" in direction:
            direction = direction.format(*direction_params)
        if directions_trace:
            direction = "substring2gibberish: " + direction
    elif corruption_function == swap_substrings:
        directions = [
            "Span {},{} was swapped with span {},{}.",
            "Mutual replacement occurred between [{},{}] and [{},{}].",
            "Undo substring swap near spans ({},{}) and ({},{}).",
            "Prepare to reverse swap of strings [{},{}], [{},{}].",
            "Two substrings in the passage were swapped.",
            "A string in the passage was replaced with a different string and vice versa.",
            "Unswap swapped substrings.",
            "One span in the passage was swapped for another span."
        ]
        corrupted, direction_params = corruption_function(passage)
        direction = random.choice(directions)
        if "{}" in direction and direction_params:
            direction = direction.format(*direction_params)
        elif not direction_params:
            return passage, "noop"
        if directions_trace:
            direction = "swap_substrings: " + direction
    elif corruption_function == adjacent_substring_swap:
        directions = [
            "The halves of span {},{} were reversed in place.",
            "Undo adjacent substring swap over interval [{},{}].",
            "String near ({},{}) was cut in half and then switched.",
            "The first half of {},{} was put in place of the second and vice versa.",
            "Reverse adjacent substring swap.",
            "Two halves of a single substring were exchanged.",
            "Detected adjacent span reversal.",
            "The left and right halves of a span swapped places."
        ]
        corrupted, direction_params = corruption_function(passage)
        direction = random.choice(directions)
        if "{}" in direction and direction_params:
            direction = direction.format(*direction_params)
        elif not direction_params:
            return passage, "noop"
        if directions_trace:
            direction = "adjacent_substring_swap: " + direction
    elif corruption_function == delete_substring:
        directions = [
            "Substring near {},{} was deleted.",
            "The span between positions {} and {} was erased.",
            "Detected removal of string near former span [{},{}].",
            "Infill deleted span near ({},{}).",
            "A string was deleted from the passage.",
            "Preparing infill of a removed substring.",
            "One of the spans in the passage is conspicuously absent.",
            "Restoring deleted string..."
        ]
        corrupted, direction_params = corruption_function(passage)
        direction = random.choice(directions)
        if "{}" in direction:
            direction = direction.format(*direction_params)
        if directions_trace:
            direction = "delete_substring: " + direction
    elif corruption_function == insert_spurious_html_xml_tag:
        directions = [
            "Spurious HTML/XML tag inserted at character position {}: {}",
            "Unnecessary tag near {} of type {}",
            "Undo insertion at {} of {} tag.",
            "At position {}, found unwanted {} tag in passage.",
            "Remove unwanted HTML tag.",
            "Found unnecessary HTML/XML tag, removing...",
            "Delete XML tag.",
            "A spurious HTML element was inserted."
        ]
        corrupted, direction_params = corruption_function(passage)
        direction = random.choice(directions)
        if "{}" in direction:
            direction = direction.format(*direction_params)
        if directions_trace:
            direction = "insert_spurious_html_xml_tag: " + direction
    elif corruption_function == delete_whitespace_character:
        directions = [
            "Deleted the {} whitespace character.",
            "The {} whitespace in the passage was removed.",
            "A whitespace was taken away at index {} over whitespace characters.",
            "Whitespace was backspaced at deletion index {}",
            "A space or newline character is missing.",
            "Detected missing whitespace in passage.",
            "Preparing to restore absent whitespace...",
            "Missing whitespace found in the text."
        ]
        corrupted, direction_params = corruption_function(passage)
        direction = random.choice(directions)
        if "{}" in direction and direction_params:
            direction = direction.format(*direction_params)
        elif not direction_params:
            return passage, "noop"
        if directions_trace:
            direction = "delete_whitespace_character: " + direction
    elif corruption_function == duplicate_word:
        directions = [
            "Word was duplicated near {} word index.",
            "{}, duplicate word detected.",
            "The {} word index has a repeated word.",
            "Two of the same word occur around index {}.",
            "Remove duplicate word.",
            "Undo repeated word in passage.",
            "A 2nd copy of the same word was inserted.",
            "Double word found in the text."
        ]
        corrupted, direction_params = corruption_function(passage)
        direction = random.choice(directions)
        if "{}" in direction and direction_params:
            direction = direction.format(*direction_params)
        elif not direction_params:
            return passage, "noop"
        if directions_trace:
            direction = "duplicate_word: " + direction
    elif corruption_function == insert_printable_ascii_character:
        directions = [
            "Random character inserted at character index {}.",
            "At char index {} a symbol was added.",
            "Remove noise symbol at position {}.",
            "Random ASCII added near {}.",
            "A printable ASCII symbol was added at a random place.",
            "Undo inserted ASCII noise symbol.",
            "A single noise character was added to the text.",
            "Detected spurious printed character in passage."
        ]
        corrupted, direction_params = corruption_function(passage)
        direction = random.choice(directions)
        if "{}" in direction:
            direction = direction.format(*direction_params)
        if directions_trace:
            direction = "insert_printable_ascii_character: " + direction
    elif corruption_function == shuffle_word_middle:
        directions = [
            "{}, word had its middle characters shuffled at word index.",
            "Inner char shuffle took place at word index {}.",
            "Letters swapped in middle of word near index {}.",
            "Inner character swap @ word index {}.",
            "Undo inside shuffle of word in passage.",
            "A word in the passage had its insides scrambled up.",
            "Shuffled random word innards.",
            "Detected a inner character anagram in the text."
        ]
        corrupted, direction_params = corruption_function(passage)
        direction = random.choice(directions)
        if "{}" in direction and direction_params:
            direction = direction.format(*direction_params)
        elif not direction_params:
            return passage, "noop"
        if directions_trace:
            direction = "shuffle_word_middle: " + direction
    elif corruption_function == swap_capitalization:
        directions = [
            "Swapped capitalization near {}.",
            "{}, changed case at this character index.",
            "Detected case swap around char index {}.",
            "Possible capitalization swap around position {}.",
            "A character had its case changed.",
            "Undo case change of random character.",
            "Possible case change found in passage.",
            "A letter changed case in the text."
        ]
        corrupted, direction_params = corruption_function(passage)
        direction = random.choice(directions)
        if "{}" in direction:
            direction = direction.format(*direction_params)
        if directions_trace:
            direction = "swap_capitalization: " + direction
    elif corruption_function == insert_punctuation:
        directions = [
            "Unnecessary punctuation inserted near char index {}.",
            "{}, spurious punctuation spotted.",
            "Unneeded punctuation added around character position {}.",
            "Position {} in the text had a random puncutation mark added to it.",
            "Found puncutation the text didn't need.",
            "Undo insertion of random punctuation mark.",
            "Remove spurious period/exclamation/etc.",
            "Detected punctuation inserted where it shouldn't be."
        ]
        corrupted, direction_params = corruption_function(passage)
        direction = random.choice(directions)
        if "{}" in direction:
            direction = direction.format(*direction_params)
        if directions_trace:
            direction = "insert_punctuation: " + direction
    elif corruption_function == adjacent_word_swap:
        directions = [
            "{}, word swapped with its neighbor.",
            "Word exchanged with its neighbor near word index {}.",
            "Undo adjacent word swap around index {}.",
            "Found word swap corruption in vicity of {}.",
            "Two adjacent words were swapped in the passage.",
            "Detected two nearby words changing places.",
            "Word swapped locations with its neighbor.",
            "Reverse word swap."
        ]
        corrupted, direction_params = corruption_function(passage)
        direction = random.choice(directions)
        if "{}" in direction and direction_params:
            direction = direction.format(*direction_params)
        elif not direction_params:
            return passage, "noop"
        if directions_trace:
            direction = "adjacent_word_swap: " + direction
    elif corruption_function == random_number_replacement:
        directions = [
            "Number #{} in passage replaced with random number.",
            "The number with n {} was randomly changed.",
            "Corrupted number at number index {}.",
            "Of the numbers in the passage the {} number has a random replacement.",
            "Detected number corruption.",
            "Undo replacement of number with a random substitute.",
            "A random number in the passage was selected and replaced with noise.",
            "One of the numbers in this text is wrong."
        ]
        corrupted, direction_params = corruption_function(passage)
        direction = random.choice(directions)
        if "{}" in direction and direction_params:
            direction = direction.format(*direction_params)
        elif not direction_params:
            return passage, "noop"
        if directions_trace:
            direction = "random_number_replacement: " + direction
    return corrupted, direction

def random_unix_timestamp_between_dates(start_date, end_date):
    # Convert the start and end dates to Unix timestamps
    start_timestamp = int(time.mktime(start_date.timetuple()))
    end_timestamp = int(time.mktime(end_date.timetuple()))

    # Generate a random timestamp within the range
    random_timestamp = random.randint(start_timestamp, end_timestamp)

    return random_timestamp

# Define the start and end dates
start_date = datetime(2024, 6, 1)
end_date = datetime(2024, 10, 1)

orientation_agendas = [
    "I notice {filename} has been corrupted. My best guess at the damage is:",
    "{filename} appears to have not saved properly. Lets think step by step to fix it.",
    "I see a number of problems with {filename} I should resolve before moving on.",
    "Aha! {filename} is clearly in a broken state. We can fix it by resolving the following issues:",
    "Looking over the observation blocks I see that {filename} is in a bad state. I'll need to fix it before I write any more.",
    "{filename} has been damaged. I can invert the following steps to repair it.",
    "There was an error while saving {filename}. I'll have to apply a patch to get the file how I want it.",
    "I'm going to need to patch {filename}. The problems I need to patch are:",
    "{filename} has the following issues:",
    "Something must have happened to {filename}, it has the following problems:",
    "Thinking step by step about {filename} I observe that...",
    "Lets list the flaws in {filename} before we fix them.",
    "I need to use the WeaveEditor to fix the problems in {filename}.",
    "{filename} has issues I can use the WeaveEditor to resolve. The issues are:",
    "I should repair {filename} with the editor tool. I need to reverse the following flaws:",
    "While the {filename} is damaged I think I can salvage it using the editor. I'll write an action to undo:",
]

action_callback_name = [
    "apply_patch",
    "undo_file_damage", "undo_data_damage", "undo_text_damage",
    "do_file_edits", "do_data_edits", "do_text_edits",
    "patch_file", "patch_data", "patch_text",
    "repair_file", "repair_data", "repair_text",
    "restore_file", "restore_data", "restore_text",
    "rescue_file", "rescue_data", "rescue_text",
    "recreate_file", "recreate_data", "recreate_text",
    "impute_file", "impute_data", "impute_text",
    "uncorrupt_file", "uncorrupt_data", "uncorrupt_text",
    "fix_file", "fix_data", "fix_text",
]

action_callback_description = [
    "Repair {filename}",
    "Fix {filename}",
    "Patch {filename}",
    "Apply patch to {filename}",
    "Attempt to fix detected flaws in {filename}",
    "Undo damage to {filename}",
    "Recreate {filename} based on inferred original contents",
    "Correct {filename} with WeaveEditor",
]

action_edit_docstrings = [
    "Use the WeaveEditor to fix the corruptions in {filename}",
    "Attempt to repair damage to {filename} using WeaveEditor.",
    "Use the editor tool to undo the corrupted spans.",
    "Perform a series of edits to repair corruptions.",
    "Execute edits to repair damage to {filename} identified in orientation.",
    "Fix {filename} using WeaveEditor commands.",
    "Repair the file by addressing list of flaws in the orientation.",
    "Translate problems listed in the orientation into line edits."
]

action_unidiff_docstrings = [
    "Use a unidiff to repair the flaws in {filename}.",
    "Tackle the corruptions discussed in the orientation with a unidiff.",
    "Apply unidiff against {filename} to address the corrupted spans.",
    "Take action to repair the problems I identified with {filename}.",
    "Compose a unidiff to restore {filename} to readable form.",
    "Use the unidiff_edit() method of WeaveEditor to fix {filename}",
    "Repair {filename} using the unidiff edit.",
    "We can restore the file using a unidiff.",
    "Callback that applies a unidiff to resolve the issues with {filename}",
    "{filename} has been damaged in a way we can fix with a unidiff.",
    "Reverse damage to {filename} using the bullet list in orientation as a checklist.",
    "Resolve the issues with {filename} by patching it with a unidiff.",
    "Patches {filename} based on the flaws I listed in the orientation stage.",
    "Patch file to reverse the damage.",
    "WeaveEditor accepts a unidiff so we can fix all the flaws in {filename} at once.",
    "Create a diff to resolve the problems seen in {filename}."
]

with open("lorem_ipsum_final.json") as infile:
    lorem_ipsum = json.load(infile)

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x22B-v0.1")

working_dir = os.getcwd()

filename_builder = FileNameBuilder('filenames.txt')
filenames = filename_builder.generate_variations()

diffs = []
pbar = tqdm(total=900)
index = 0
while lorem_ipsum:
    text = lorem_ipsum.pop()["text"]
    text_herring1 = lorem_ipsum.pop()["text"]
    text_herring2 = lorem_ipsum.pop()["text"]
    corrupted_passage = text
    directions_trace = random.randint(0,1)
    log = ""
    roll = random.randint(3,10)
    for i in range(roll):
        corrupted_passage, logline = do_corruption(corrupted_passage,
                                                   lorem_ipsum,
                                                   directions_trace)
        if logline == "noop":
            continue
        log += (logline + "\n")
    if not log:
        continue

    orientation_timestamp = random_unix_timestamp_between_dates(start_date, end_date)
    action_timestamp = orientation_timestamp + random.randint(60, 300)
    
    text_corrupted_filename = random.choice(filenames)
    text_herring1_filename = random.choice(filenames)
    excluded = [text_corrupted_filename,]
    while text_herring1_filename in excluded:
        text_herring1_filename = random.choice(filenames)
    text_herring2_filename = random.choice(filenames)
    excluded = [text_corrupted_filename, text_herring1_filename]
    while text_herring2_filename in excluded:
        text_herring2_filename = random.choice()

    diff = difflib.unified_diff(corrupted_passage.splitlines(),
                                text.splitlines(), lineterm='\n')
    diff_lines = list(diff)

    # Mock agent object
    agent = type('Agent', (object,), {'tools': [], 'add_observation_view': lambda self, x, y: None, 'remove_observation_view': lambda self, x: None})()
    if os.path.exists(text_corrupted_filename):
        raise ValueError
    with open(text_corrupted_filename, "w") as outfile:
        outfile.write(text)
        outfile.flush()
    editor = WeaveEditor(agent, text_corrupted_filename)
    original_lines = editor.file_content
    with open(text_corrupted_filename, "w") as outfile:
        outfile.write(corrupted_passage)
        outfile.flush()
    editor.load_file(text_corrupted_filename)
    corrupted_lines = editor.file_content
    editor.unidiff_edit(diff_lines)
    assert editor.file_content == original_lines
    os.remove(text_corrupted_filename)
    
    diffs.append(
        {
            "text_corrupted":corrupted_passage,
            "text_herring1":text_herring1,
            "text_herring2":text_herring2,
            "text_corrupted_filename": text_corrupted_filename,
            "text_herring1_filename": text_herring1_filename,
            "text_herring2_filename": text_herring2_filename,
            "orientation_timestamp":orientation_timestamp,
            "orientation_agenda":random.choice(orientation_agendas),
            "operations":log,
            "action_timestamp":action_timestamp,
            "action_callback_name":random.choice(action_callback_name),
            "action_edit_docstring":random.choice(action_edit_docstrings),
            "action_unidiff_docstring":random.choice(action_unidiff_docstrings),
            "diff":diff_lines,
            "edits":parse_diff(diff_lines),
            "action_callback_description":random.choice(action_callback_description),
            "text_clean":text
        }
    )

    print(i, roll, log)
    index += 1
    pbar.update(1)


# Make train and val split

random.shuffle(diffs)

val_size = round(len(diffs) / 10) 
train_size = len(diffs) - val_size

os.chdir(working_dir)

with open("train.json", "w") as outfile:
    json.dump(diffs[:train_size], outfile)

with open("val.json", "w") as outfile:
    json.dump(diffs[train_size:], outfile)
