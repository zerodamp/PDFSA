from PyPDF2 import PdfReader
from nltk.tokenize import sent_tokenize
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.nlp.tokenizers import Tokenizer
from rake_nltk import Rake  # Rake'yi ekledik

def anahtar_kelime_cikar(metin, maksimum_kelime_sayisi=100):
    # Anahtar kelime çıkarma işlemi
    r = Rake()
    r.extract_keywords_from_text(metin)
    anahtar_kelimeler = r.get_ranked_phrases()[:maksimum_kelime_sayisi]
    return anahtar_kelimeler

def pdf_metni_oku(pdf_yolu):
    try:
        # PDF'den metin çıkarma işlemi
        pdf = PdfReader(pdf_yolu)
        metin = ""
        for sayfa in pdf.pages:
            metin += sayfa.extract_text()
        return metin
    except Exception as e:
        print(f"Hata: {e}")
        return None

def anlam_kontrolu(ozet, metin):
    # Anlam kontrolü
    anahtar_kelimeler_ozet = set(ozet.split())
    kelimeler_metin = set(metin.split())

    if not anahtar_kelimeler_ozet.issubset(kelimeler_metin):
        return False
    return True

def metni_analiz_et_textrank(metin, cümle_sayisi=5):
    # Textrank algoritması ile özetleme işlemi
    parser = PlaintextParser.from_string(metin, Tokenizer("turkish"))
    summarizer = TextRankSummarizer()
    ozet = summarizer(parser.document, cümle_sayisi)

    return ' '.join([str(cumle) for cumle in ozet])

def metni_analiz_et(metin, maksimum_karakter=500):
    # Özel özetleme işlemi (örneğin, metni ilk 500 karakterle sınırlamak)
    return metin[:maksimum_karakter]

# Kullanıcıdan PDF dosyasının yolunu al
pdf_yolu = input("PDF dosyasının yolunu girin: ")

# PDF'den metin çıkar ve özetle
pdf_metni = pdf_metni_oku(pdf_yolu)
if pdf_metni:
    # Cümle sayısını belirle
    cumleler = sent_tokenize(pdf_metni)
    toplam_cumle_sayisi = len(cumleler)

    # Özetleme işlemi (maksimum 500 karakter)
    ozet_textrank = metni_analiz_et_textrank(pdf_metni)
    print("\nTextrank Özet:")
    print(ozet_textrank)

    # Anlam kontrolü yap
    if not anlam_kontrolu(ozet_textrank, pdf_metni):
        print("\nTextrank Özet anlamsız. Diğer algoritmalarla yeniden özetleniyor...")

        # Diğer algoritmalarla yeniden özetleme işlemi (maksimum 500 karakter)
        ozet = metni_analiz_et(pdf_metni, maksimum_karakter=500)
        print("\nYeniden Özet:")
        print(ozet)
    else:
        print("\nTextrank Özet anlamlı.")

else:
    print("PDF'den özet çıkarma işlemi başarısız.")
