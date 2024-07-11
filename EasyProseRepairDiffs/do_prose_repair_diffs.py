import random
import string
import json
import re
import os
import tempfile
import git
from git import Repo
import subprocess
from diff_match_patch import diff_match_patch
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

    # Replace the first occurrence of each substring with the other substring
    corrupted_passage = passage.replace(substring1, substring2, 1)
    corrupted_passage = corrupted_passage.replace(substring2, substring1, 1)

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
    whitespace_chars = [c for c in passage if c.isspace()]

    # If there are no whitespace characters, return the original passage
    if not whitespace_chars:
        return passage, tuple()

    # Choose a random whitespace character to delete
    whitespace_char = random.choice(whitespace_chars)

    # Choose a random position in the passage to delete the whitespace character
    delete_index = random.randint(0, passage.count(whitespace_char) - 1)

    # Delete the whitespace character at the chosen position
    corrupted_passage = passage.replace(whitespace_char, '', delete_index)

    return corrupted_passage, (delete_index,)


def duplicate_word(passage):
    # Split the passage into a list of words
    words = passage.split()

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
    words = passage.split()

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
    words = passage.split()

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


def get_git_diff(file_content, changed_content):
    # Create a temporary directory
    temp_dir = tempfile.TemporaryDirectory()

    # Initialize a git repository in the temporary directory
    repo_dir = os.path.join(temp_dir.name, 'repo')
    os.mkdir(repo_dir)
    os.chdir(repo_dir)
    subprocess.run(['git', 'init'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Create a file and write the initial content
    file_path = os.path.join(repo_dir, 'test.txt')
    with open(file_path, 'w') as f:
        f.write(file_content)

    # Commit the initial file
    subprocess.run(['git', 'add', 'test.txt'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(['git', 'commit', '-m', 'Initial commit'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Overwrite the file with the changed content
    with open(file_path, 'w') as f:
        f.write(changed_content)

    # Commit the changed file
    subprocess.run(['git', 'add', 'test.txt'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(['git', 'commit', '-m', 'Changed commit'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Get the diff between the two commits
    diff = subprocess.run(['git', 'diff', 'HEAD~1..HEAD'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    # Clean up the temporary directory
    temp_dir.cleanup()

    # Print the diff
    return diff.stdout.decode()

    
def get_gnu_diff(file_content, changed_content):
    # Create two temporary files
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as orig_file, \
         tempfile.NamedTemporaryFile(mode='w', delete=False) as changed_file:
        # Write the original and changed content to the files
        orig_file.write(file_content)
        changed_file.write(changed_content)

    # Use diff to get the differences
    result = subprocess.run(['diff', '-u', orig_file.name, changed_file.name], stdout=subprocess.PIPE)

    # Clean up the temporary files
    os.remove(orig_file.name)
    os.remove(changed_file.name)

    # Print the diff
    return result.stdout.decode()


def get_patch_match_diff(file_content, changed_content):
    # Create a diff-match-patch object
    dmp = diff_match_patch()

    # Get a list of diffs
    diffs = dmp.diff_main(file_content, changed_content)

    # Convert the diffs to patches
    patches = dmp.patch_make(file_content, changed_content, diffs)

    # Get a textual representation of the patches
    diff_text = dmp.patch_toText(patches)
    
    # Return the diff as a string
    return diff_text


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

gnudiff_instructions = [
    "Repair the following context window by diagnosing its flaws and writing a diff in GNU diff format to fix them.",
    "A series of corruptions have been applied to the following passage. Find the problems and repair them with a GNU diff.",
    "Write a GNU diff to repair the problems in this text.",
    "List the problems in this passage and author a GNU diff which fixes them.",
    "Think step by step to diagnose the issues with this text then amend it with a GNU diff.",
    "i need you to fix up this writing for me using a GNU diff",
    "Use a GNU diff to repair the following passage.",
    "Produce a diff in GNU format to patch the problems in this prose."
]

gitdiff_instructions = [
    "The following text has been intentionally corrupted. Use a git diff to repair it.",
    "Use your expertise to diagnose the problems with this passage and provide a git diff to fix them.",
    "The following text has been tampered with. Use a git diff to undo the damage.",
    "This text has been altered in multiple ways. Write a git diff to repair it.",
    "Diagnose the issues with this text and provide a git diff to repair them.",
    "Use your knowledge of git diffs to repair the following text.",
    "Use a git diff to correct the errors in this passage.",
    "This text has been damaged. Use a git diff to diagnose and repair the issues."
]

dmp_instructions = [
    "The following text has been corrupted in a way that the diff-match-patch format can fix.",
    "Using diff-match-patch commands, repair the following text.",
    "List the problems in this passage and write a diff-match-patch patch to fix them.",
    "The text below has been tampered with. Use a diff_match_patch to fix it.",
    "Diagnose the corruptions in this text and use a diff_match_patch to fix them all.",
    "Use your knowledge of diff-match-patch format to diagnose and repair the errors in this passage.",
    "Find the issues with this passage and write a diff-match-patch to repair them.",
    "Infer the issues with this text and generate a diff_match_patch_patch_toText formatted patch to repair it."
]
    
with open("lorem_ipsum_final.json") as infile:
    lorem_ipsum = json.load(infile)

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x22B-v0.1")

working_dir = os.getcwd()

diffs = []
for i, li in tqdm(enumerate(lorem_ipsum)):
    text, score = li["text"], li["score"]
    text = tokenizer.decode(tokenizer(text)["input_ids"][:1200], skip_special_tokens=True)
    corrupted_passage = text
    directions_trace = random.randint(0,1)
    log = ""
    roll = random.randint(1,10)
    for i in range(roll):
        corrupted_passage, logline = do_corruption(corrupted_passage,
                                                   lorem_ipsum,
                                                   directions_trace)
        log += (logline + "\n")
    gnudiff = get_gnu_diff(corrupted_passage, text)
    gitdiff = get_git_diff(corrupted_passage, text)
    dmpdiff = get_patch_match_diff(corrupted_passage, text)

    diffs.append(
        {
            "gnudiff_instruction":random.choice(gnudiff_instructions),
            "gitdiff_instruction":random.choice(gitdiff_instructions),
            "dmpdiff_instruction":random.choice(dmp_instructions),
            "text_corrupted":corrupted_passage,
            "operations":log,
            "gnudiff":gnudiff,
            "gitdiff":gitdiff,
            "dmpdiff":dmpdiff,
            "text_clean":text
        }
    )

    print(i, roll, log)


# Make train and val split

random.shuffle(diffs)

val_size = round(len(diffs) / 10) 
train_size = len(diffs) - val_size

os.chdir(working_dir)

with open("train.json", "w") as outfile:
    json.dump(diffs[:train_size], outfile)

with open("val.json", "w") as outfile:
    json.dump(diffs[train_size:], outfile)
