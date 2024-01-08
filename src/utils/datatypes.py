from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union


class Language(Enum):
    EN = "en"


@dataclass(frozen=True, eq=True)
class Span:
    start: int
    end: int
    label: str
    def to_json(self):
        return {"start": self.start, "end": self.end, "label": self.label}

    @staticmethod
    def from_json(data):
        return Span(start=int(data["start"]), end=int(data["end"]), label=data["label"])

    @staticmethod
    def from_tuple(data):
        return Span(start=int(data[0]), end=int(data[1]), label=data[2])


@dataclass()
class Instance:
    context: str
    question: str
    answer: Optional[str]
    entity_spans: Optional[Union[List[Span], List[Tuple[int, int, str]]]]
    entity_values: Optional[Dict[str, List[str]]]
    img_idx: str
    def __str__(self):

        if self.answer is not None:
            return self.context + " " + self.question + " " + self.answer

        return self.context + " " + self.question


class AnswerFormatter:
    available_languages = [Language.EN.value]
    patterns = {
        Language.EN.value: "{0} is {1} {2}",  # value is a/and label
    }
    
    @classmethod
    def from_values(
        cls, entity_values: Optional[Dict[str, List[str]]], language: str = "en"
    ) -> Optional[str]:
        if language not in cls.available_languages:
            raise ValueError(
                f"Expected language to be one of {cls.available_languages}"
            )
        if entity_values is None:
            return None
        answers = []
        for entity_label, values in entity_values.items():
            if entity_label.lower()[0] in ["a", "e", "u", "o"]:
                answers.extend(
                    [
                        cls.patterns[language].format(value, "an", entity_label)
                        for value in values
                    ]
                )
            else:
                answers.extend(
                    [
                        cls.patterns[language].format(value, "a", entity_label)
                        for value in values
                    ]
                )
        answer = ", ".join(answers) + "."

        return answer

    @classmethod
    def from_spans(
        cls,
        context: str,
        entity_spans: List[Tuple[int, int, str]],
        language: str = "en",
    ) -> str:
        answers = []
        if language not in cls.available_languages:
            raise ValueError(
                f"Expected language to be one of {cls.available_languages}"
            )
        for span in entity_spans:
            start, end, label = span[0], span[1], span[2]
            value = context[start:end]
            if value.lower().startswith("a"):
                answers.append(cls.patterns[language].format(value, "an", label))
            else:
                answers.append(cls.patterns[language].format(value, "a", label))
        answer = ", ".join(answers) + "."

        return answer


class Preffix(Enum):
    CONTEXT = "Sentence: "
    INSTRUCTION = "Instruction: "
    OPTIONS = "Options: "
    ANSWER = "Answer: "


class TaskType(Enum):
    ETCT = "ETCT"
    ETT = "ETT"
    EET = "EET"