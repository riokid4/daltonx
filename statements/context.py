from datetime import datetime
from typing import Optional


__all__ = ['Context', 'AISafetyContext', 'RefContext', 'TimeContext']


class Context(object):
    """An abstract class for Contexts."""
    @classmethod
    def from_json(cls, jd):
        context_type = jd.get('type')
        if context_type in ('ai_safety', 'world'):
            return AISafetyContext.from_json(jd)
        else:
            raise ValueError('Unknown context type %s' % context_type)


class RefContext(object):
    """An object representing a context with a name and references.

    Parameters
    ----------
    name : Optional[str]
        The name of the given context.
    db_refs : Optional[dict]
        A dictionary where each key is a namespace and each value is an
        identifier in that namespace.
    """
    def __init__(self, name=None, db_refs=None):
        self.name = name
        self.db_refs = {} if db_refs is None else db_refs

    def __eq__(self, other):
        return (self.name == getattr(other, 'name', None)) and \
               (self.db_refs == getattr(other, 'db_refs', None))

    def __ne__(self, other):
        return not self.__eq__(other)

    def to_json(self):
        jd = {'name': self.name, 'db_refs': self.db_refs}
        return jd

    @classmethod
    def from_json(cls, jd):
        if not jd:
            return None
        rc = cls(name=jd.get('name'), db_refs=jd.get('db_refs'))
        return rc

    def __str__(self):
        pieces = []
        if self.name:
            pieces.append('name="%s"' % self.name)
        if self.db_refs:
            pieces.append('db_refs=%s' % self.db_refs)
        args = ', '.join(pieces)
        return 'RefContext(%s)' % args

    def __repr__(self):
        return self.__str__()


class TimeContext(object):
    """An object representing the time context of a Statement.

    Parameters
    ----------
    text : Optional[str]
        A string representation of the time constraint, typically as seen in text.
    start : Optional[datetime]
        A `datetime` object representing the start time.
    end : Optional[datetime]
        A `datetime` object representing the end time.
    duration : Optional[int]
        The duration of the time constraint in seconds.
    """
    def __init__(self, text=None, start=None, end=None, duration=None):
        self.text = text
        self.start = start
        self.end = end
        self.duration = duration

    def refinement_of(self, other):
        if self.start and other.start:
            if self.start <= other.start:
                return False
            elif self.end and other.end:
                if self.end >= other.end:
                    return False
            elif self.end and not other.end:
                return True
            return True
        elif self.start and not other.start:
            return True
        return False

    def __eq__(self, other):
        return (self.text == getattr(other, 'text', None)) and \
               (self.start == getattr(other, 'start', None)) and \
               (self.end == getattr(other, 'end', None)) and \
               (self.duration == getattr(other, 'duration', None))

    def __ne__(self, other):
        return not self.__eq__(other)

    def to_json(self):
        def date_to_str(date):
            if date is None:
                return None
            elif isinstance(date, str):
                return date
            else:
                return date.strftime('%Y-%m-%dT%H:%M')
        jd = {'text': self.text,
              'start': date_to_str(self.start),
              'end': date_to_str(self.end)}
        if self.duration is not None:
            jd['duration'] = self.duration
        return jd

    @classmethod
    def from_json(cls, jd):
        def parse_date(date_str):
            if not date_str:
                return None
            try:
                return datetime.strptime(date_str, '%Y-%m-%dT%H:%M')
            except ValueError:
                return date_str

        text = jd.get('text')
        start = parse_date(jd.get('start'))
        end = parse_date(jd.get('end'))
        duration = jd.get('duration')
        return cls(text=text, start=start, end=end, duration=duration)

    def __str__(self):
        pieces = []
        if self.text:
            pieces.append('text="%s"' % self.text)
        if self.start:
            pieces.append('start=%s' % self.start)
        if self.end:
            pieces.append('end=%s' % self.end)
        if self.duration is not None:
            pieces.append('duration=%s' % self.duration)
        args = ', '.join(pieces)
        return 'TimeContext(%s)' % args

    def __repr__(self):
        return self.__str__()


class AISafetyContext(Context):
    """An object representing the operational, temporal, and model context of an AI safety Statement.

    Parameters
    ----------
    time : Optional[TimeContext]
        A TimeContext object representing the temporal context of the Statement.
    environment : Optional[RefContext]
        The testing, benchmark, deployment, or simulation environment (e.g. RL sandbox, production API).
    model : Optional[RefContext]
        The AI model or system under evaluation (e.g. GPT-4, Claude 3, Llama 3).
    geo_location : Optional[RefContext]
        The geographical or jurisdictional context (e.g. EU AI Act, US, Global).
    """
    def __init__(self, time=None, environment=None, model=None, geo_location=None):
        self.time = time
        self.environment = environment
        self.model = model
        self.geo_location = geo_location

    def __eq__(self, other):
        return (self.time == getattr(other, 'time', None)) and \
               (self.environment == getattr(other, 'environment', None)) and \
               (self.model == getattr(other, 'model', None)) and \
               (self.geo_location == getattr(other, 'geo_location', None))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __bool__(self):
        return (self.time is not None or
                self.environment is not None or
                self.model is not None or
                self.geo_location is not None)

    @classmethod
    def from_json(cls, jd):
        time_entry = jd.get('time')
        time = TimeContext.from_json(time_entry) if time_entry else None
        env_entry = jd.get('environment')
        environment = RefContext.from_json(env_entry) if env_entry else None
        model_entry = jd.get('model')
        model = RefContext.from_json(model_entry) if model_entry else None
        geo_entry = jd.get('geo_location')
        geo_location = RefContext.from_json(geo_entry) if geo_entry else None
        return cls(time=time, environment=environment, model=model,
                   geo_location=geo_location)

    def to_json(self):
        jd = {'type': 'ai_safety',
              'time': self.time.to_json() if self.time else None,
              'environment': self.environment.to_json() if self.environment else None,
              'model': self.model.to_json() if self.model else None,
              'geo_location': (self.geo_location.to_json()
                               if self.geo_location else None)}
        return jd

    def __str__(self):
        pieces = []
        if self.time:
            pieces.append('time=%s' % self.time)
        if self.environment:
            pieces.append('environment=%s' % self.environment)
        if self.model:
            pieces.append('model=%s' % self.model)
        if self.geo_location:
            pieces.append('geo_location=%s' % self.geo_location)
        args = ', '.join(pieces)
        return 'AISafetyContext(%s)' % args

    def __repr__(self):
        return self.__str__()
