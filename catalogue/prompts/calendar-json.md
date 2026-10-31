Format: a JSON array of calendar events, one object per event the speaker described, with exactly these keys:

[{"title": str, "date_text": str|null, "start_time": str|null, "end_time": str|null, "location": str|null, "attendees": [str], "description": str|null}]

- title: short, from the speaker's words ("Dentist", "Lunch with Sarah").
- date_text: the date as the speaker said it ("next Tuesday", "March 12"). Do not convert relative dates.
- start_time / end_time: 24-hour "HH:MM" when the speaker gave a time; null otherwise. Never guess a duration.
- attendees: names as spoken; [] if none. Never invent email addresses.
- description: anything else the speaker wanted noted about that event; null if nothing.
- Output valid JSON only, compact or indented, nothing before or after it.
