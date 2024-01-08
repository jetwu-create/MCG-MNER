from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
from utils.datatypes import Preffix, Span, Instance, AnswerFormatter


class InstanceFormatter(ABC):
    @abstractmethod
    def format_instance(
        self,
        context,
        entity_values,
        entity_spans,
        instruction,
        options,
    ):

        raise NotImplementedError

class ETCT_(InstanceFormatter):
    def format_instance(
        self,
        context,
        entity_values,
        entity_spans,
        img_idx,
        instruction,
        options,
    ):
        question = Preffix.INSTRUCTION.value + instruction
        answer = None
        if entity_values is not None:
            answers = []
            opt_key = list(entity_values.keys())
            opt_value = list(entity_values.values())
            if len(opt_key) == 1:
                if opt_key[0] == "PER":
                    answers.extend(f"{opt_key[0]}: {len(opt_value)}, LOC: 0, ORG: 0, MISC: 0.")
                elif opt_key[0] == "LOC":
                    answers.extend(f"PER: 0, {opt_key[0]}: {len(opt_value)}, ORG: 0, MISC: 0.")
                elif opt_key[0] == "ORG":
                    answers.extend(f"PER: 0, LOC: 0, {opt_key[0]}: {len(opt_value)}, MISC: 0.")
                else:
                    answers.extend(f"PER: 0, LOC: 0, ORG: 0, {opt_key[0]}: {len(opt_value)}.")
            elif len(opt_key) == 0:
                answers.extend(f"PER: 0, LOC: 0, ORG: 0, MISC: 0.")
            else:
                cout_1, cout_2, cout_3, cout_4 = 0, 0, 0, 0
                for i in range(len(opt_key)):
                    if opt_key[i] == "PER":
                        cout_1 += len(opt_value[i])
                    elif opt_key[i] == "LOC":
                        cout_2 += len(opt_value[i])
                    elif opt_key[i] == "ORG":
                        cout_3 += len(opt_value[i])
                    else:
                        cout_4 += len(opt_value[i])
                        
                answers.extend(f"PER: {cout_1}, LOC: {cout_2}, ORG: {cout_3}, MISC: {cout_4}.")
                    
            answer = "".join(answers)

        if entity_spans is not None:
            entity_spans = [
                Span.from_json(span)
                for span in entity_spans
                if not isinstance(span, Span)
            ]
        instance = Instance(
            context=Preffix.CONTEXT.value + context,
            question=question,
            answer=answer,
            entity_spans=entity_spans,
            entity_values=entity_values,
            img_idx=img_idx,
        )
        
        return instance


class ETT_(InstanceFormatter):
    def format_instance(
        self,
        context,
        entity_values,
        entity_spans,
        img_idx,
        instruction,
        options,
    ):
        entity_values_total = None
        if entity_values is not None:
            entity_values_total = []
            for values in entity_values.values():
                entity_values_total.extend(values)
        instruction = (
            Preffix.INSTRUCTION.value
            + instruction
            + ": "
            + ", ".join(entity_values_total)
        )
        options_str = Preffix.OPTIONS.value + ", ".join(options)
        question = instruction + " " + options_str
        if entity_spans is not None:
            entity_spans = [
                Span.from_json(span)
                for span in entity_spans
                if not isinstance(span, Span)
            ]
        instance = Instance(
            context=Preffix.CONTEXT.value + context,
            question=question,
            answer=AnswerFormatter.from_values(entity_values),
            entity_spans=entity_spans,
            entity_values=entity_values,
            img_idx=img_idx,
        )

        return instance


class EET_():
    def format_instance(
        self,
        context,
        entity_values,
        entity_spans,
        img_idx,
        instruction,
        options,
    ):
        instruction = Preffix.INSTRUCTION.value + instruction
        options_joined = ", ".join(options)
        options_string = Preffix.OPTIONS.value + options_joined
        question = instruction + " " + options_string
        if entity_spans is not None:
            entity_spans = [
                Span.from_json(span)
                for span in entity_spans
                if not isinstance(span, Span)
            ]
        instance = Instance(
            context=Preffix.CONTEXT.value + context,
            question=question,
            answer=AnswerFormatter.from_values(entity_values),
            entity_values=entity_values,
            entity_spans=entity_spans,
            img_idx=img_idx,
        )

        return instance