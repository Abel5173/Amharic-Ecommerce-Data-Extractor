# import re
# def clean_amharic_text(text):
#     if not text:
#         return ""

#     # Keep Amharic (U+1200–U+137F), English letters, digits, +, @, space
#     allowed_pattern = r"[^\u1200-\u137F\u0041-\u005A\u0061-\u007A0-9@+\s]"
#     text = re.sub(allowed_pattern, " ", text)

#     # Normalize multiple spaces
#     text = re.sub(r"\s+", " ", text)

#     return text.strip()
#

print("Hello World")
