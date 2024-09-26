class FileNameBuilder:
    def __init__(self, filename_list):
        self.filename_list = filename_list
        self.case_options = ['lowercase', 'Capital Case', 'UPPERCASE']
        self.separator_options = ['_', '']
        self.extension_options = ['.txt', '.md']

    def read_filenames(self):
        with open(self.filename_list, 'r') as file:
            filenames = file.read().splitlines()
        return filenames

    def split_words(self, filename):
        if '_' in filename:
            return filename.split('_')
        else:
            return [filename]

    def apply_case(self, words, case_option):
        if case_option == 'lowercase':
            return [word.lower() for word in words]
        elif case_option == 'Capital Case':
            return [word.capitalize() for word in words]
        elif case_option == 'UPPERCASE':
            return [word.upper() for word in words]

    def generate_variations(self):
        filenames = self.read_filenames()
        variations = []
        for filename in filenames:
            words = self.split_words(filename)
            for case_option in self.case_options:
                cased_words = self.apply_case(words, case_option)
                if len(words) > 1:
                    for separator in self.separator_options:
                        separated_words = separator.join(cased_words)
                        for extension in self.extension_options:
                            variations.append(separated_words + extension)
                else:
                    for extension in self.extension_options:
                        variations.append(cased_words[0] + extension)
        return variations

if __name__ == "__main__":
    # Example usage
    filename_builder = FileNameBuilder('filenames.txt')
    variations = filename_builder.generate_variations()
    for variation in variations:
        print(variation)
