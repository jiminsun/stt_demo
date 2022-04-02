import enum
from difflib import get_close_matches


class IntentType(enum.IntEnum):
    """Type of intent"""
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3
    NORTHEAST = 4
    NORTHWEST = 5
    SOUTHEAST = 6
    SOUTHWEST = 7
    RUNWAY_8 = 8
    RUNWAY_26 = 9
    UNKNOWN = 10


class TextToIntent:
    def __init__(self):
        self.vocab = {w.rstrip() for w in open('vocab.txt', 'r').readlines()}

    def postprocess(self, text):
        processed = []
        for w in text.split():
            if w in self.vocab:
                processed.append(w)
            else:
                try:
                    processed.append(get_close_matches(w, self.vocab)[0])
                except:
                    processed.append(w)
        output = ' '.join(processed)
        return output

    def predict(self, text):
        text = self.postprocess(text)
        intent = self.match_intent_pattern(text)
        return intent

    def match_intent_pattern(self, text):
        if 'depart' in text:
            if 'north' in text:
                if 'east' in text:
                    return IntentType.NORTHEAST
                elif 'west' in text:
                    return IntentType.NORTHWEST
                else:
                    return IntentType.NORTH
            elif 'south' in text:
                if 'east' in text:
                    return IntentType.SOUTHEAST
                elif 'west' in text:
                    return IntentType.SOUTHWEST
                else:
                    return IntentType.SOUTH
            elif 'east' in text:
                return IntentType.EAST
            elif 'west' in text:
                return IntentType.WEST
            else:
                return IntentType.UNKNOWN
        else:
            # landing
            if 'eight' in text:
                return IntentType.RUNWAY_8
            elif 'two six' in text:
                return IntentType.RUNWAY_26
            else:
                return IntentType.UNKNOWN


if __name__ == '__main__':
    test_sentence = 'butler traffic skyhawk seven three seven is on the go runway two six left closed pattern butler traffic'
    intent_model = TextToIntent()
    print("=== [INPUT]", test_sentence)
    print("=== [PREDICTED INTENT]", intent_model.predict(test_sentence))