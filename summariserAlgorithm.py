from PyPDF2 import PdfReader
from nltk.tokenize import sent_tokenize
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.nlp.tokenizers import Tokenizer
from rake_nltk import Rake  # Added Rake

def extract_keywords(text, max_word_count=100):
    # Keyword extraction
    r = Rake()
    r.extract_keywords_from_text(text)
    keywords = r.get_ranked_phrases()[:max_word_count]
    return keywords

def read_pdf_text(pdf_path):
    try:
        # Extract text from PDF
        pdf = PdfReader(pdf_path)
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error: {e}")
        return None

def check_meaning(summary, text):
    # Meaning check
    summary_keywords = set(summary.split())
    text_keywords = set(text.split())

    if not summary_keywords.issubset(text_keywords):
        return False
    return True

def analyze_text_textrank(text, sentence_count=5, language='english'):
    # Summarize text using TextRank algorithm
    parser = PlaintextParser.from_string(text, Tokenizer(language))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, sentence_count)

    return ' '.join([str(sentence) for sentence in summary])

def analyze_text(text, max_characters=500):
    # Custom summarization (e.g., limiting the text to the first 500 characters)
    return text[:max_characters]

# Supported languages
supported_languages = ['english', 'turkish', 'spanish', 'russian', 'german', 'french', 'arabic']  # Add more languages as needed

# Prompt user to select a language
print("Supported languages: ", ', '.join(supported_languages))
selected_language = input("Select a language: ").lower()

# Validate selected language
if selected_language not in supported_languages:
    print("Invalid language selection. Defaulting to English.")
    selected_language = 'english'

# Get the path of the PDF file from the user
pdf_path = input("Enter the path of the PDF file: ")

# Extract text from PDF and summarize
pdf_text = read_pdf_text(pdf_path)
if pdf_text:
    # Determine the number of sentences
    sentences = sent_tokenize(pdf_text)
    total_sentence_count = len(sentences)

    # Summarization (maximum 500 characters)
    textrank_summary = analyze_text_textrank(pdf_text, language=selected_language)
    print("\nSummary:")
    print(textrank_summary)

    # Meaning check
    if not check_meaning(textrank_summary, pdf_text):
        print("\nSummary is meaningless. Re-summarizing with other algorithms...")

        # Re-summarize with other algorithms (maximum 500 characters)
        summary = analyze_text(pdf_text, max_characters=500)
        print("\nRe-Summary:")
        print(summary)
    else:
        print("\nSummary is meaningful.")

else:
    print("Failed to extract text from the PDF.")
