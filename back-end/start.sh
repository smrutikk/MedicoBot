#!/bin/bash
sed -i '/"cobol": Language.COBOL,/d' \
.venv/lib/python3.11/site-packages/langchain_community/document_loaders/parsers/language/language_parser.py
python bot.py
