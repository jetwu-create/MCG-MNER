from utils.datatypes import Preffix, Span
import re


class PredictionSpanFormatter:
    
    answer_templates = ['is an', 'is a']
    def format_answer_spans(
        self,
        context,
        prediction,
        options,
    ):
        entity_spans = []
        source_sentence = context.lstrip(Preffix.CONTEXT.value)
        
        prediction = prediction.strip(
            '.'
        )
        prediction_parts = prediction.split(',')
        
        for prediction_part in prediction_parts:
            spans = self._get_span_from_part(prediction_part, source_sentence)
            if spans is None:
                continue
            
            spans = [span for span in spans if span.label in options]
            entity_spans.extend(spans)
            
        return entity_spans
    
    def _get_span_from_part(
        self,
        prediction_part,
        source_sentence,
    ):
        if not any([template in prediction_part for template in self.answer_templates]):
            return None
    
        for answer_template in self.answer_templates:
            _prediction_part = prediction_part.split(answer_template, maxsplit=2)

            if len(_prediction_part) != 2:
                continue

            value, label = _prediction_part[0], _prediction_part[1]
            value = value.strip(" ").rstrip(" ")
            label = label.strip(" ").rstrip(" ")

            try:
                matches = list(re.finditer(value, source_sentence))
            except re.error:  # unbalanced parenthesis at position
                return None

            if len(matches) == 0:
                return None

            spans = []

            for match in matches:

                start = match.start()
                end = match.end()
                span = Span(start=start, end=end, label=label)
                spans.append(span)

            return spans
        
        return None