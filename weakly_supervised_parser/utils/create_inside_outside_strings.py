class InsideOutside:
    def __init__(self, sentence):
        self.sentence = sentence.split()
        self.sentence_length = len(self.sentence)

    def calculate_inside(self, idx_start, idx_end):
        # get inside string
        return self.sentence[idx_start:idx_end]

    def calculate_outside(self, idx_start, idx_end):
        # get outside string
        if idx_start == 0 and idx_end == self.sentence_length:
            left_outside = ["<s>"]  # bos_token roberta   # ["[UNK]"]
            right_outside = ["</s>"]  # eos_token roberta  # ["[UNK]"]
        elif idx_start == 0:
            left_outside = ["<s>"]  # ["[UNK]"]
            right_outside = self.sentence[idx_end:]
        elif idx_end == self.sentence_length:
            left_outside = self.sentence[:idx_start]
            right_outside = ["</s>"]  # ["[UNK]"]
        else:
            left_outside = self.sentence[:idx_start]
            right_outside = self.sentence[idx_end:]
        return left_outside, right_outside

    def create_inside_outside_matrix(self, ngram):
        i, j = ngram[0][0], ngram[0][-1]
        inside_string = self.calculate_inside(i, j)
        outside_string = self.calculate_outside(i, j)
        output_dict = {
            "span": ngram[0],
            "inside_string": " ".join(inside_string),
            "left_outside_string": " ".join(outside_string[0]),
            "right_outside_string": " ".join(outside_string[-1]),
        }
        inside_string_template = output_dict["inside_string"]
        outside_string_template = (
            output_dict["left_outside_string"].split()[-1] + " " + "<mask>" + " " + output_dict["right_outside_string"].split()[0]
        )
        return output_dict, inside_string_template, outside_string_template
