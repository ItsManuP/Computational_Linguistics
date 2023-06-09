import nltk
from nltk import bigrams
from nltk import FreqDist
from nltk.stem import WordNetLemmatizer
import sys
from nltk.tokenize import RegexpTokenizer
import collections



def letturafile(file):
    with open(file, "r", encoding="utf-8") as file:
        file_contenuto = file.read()
    return file_contenuto


def get_tokens(testo):
    sentences = nltk.tokenize.sent_tokenize(testo)
    lista_tokens = []
    for sentence in sentences:
        sentence_tokens = nltk.tokenize.word_tokenize(sentence)
        lista_tokens = lista_tokens + sentence_tokens
    return lista_tokens


def LunghezzaMediaFrasi(frasi, numero_tokens):
    length_frasi = len(frasi)
    # Divido il numero dei tokens per la length delle frasi
    media = (numero_tokens*1.0/length_frasi*1.0)
    return media  # restituisco la media

def LunghezzaMediaToken(file_tokens,file_numero_tokens):
    lettere = 0
    for x in file_tokens:
        lettere = lettere + len(x)
        media = (lettere*1.0/file_numero_tokens*1.0)
    return media


# Questa funzione mi conteggia le parole che appaiono solamente una volta nel testo, ho sfruttato una list comprehension.
# itera sulla parola e controllo che il contatore della parola sia uguale a 1, in quel caso la includo nella lista hapax.
# counter contiene un conteggio  delle occorrenze di ogni parola nel testo.
def hapax_contatore(tokens):
    counter = collections.Counter(tokens)
    hapax = [word for word, count in counter.items() if count == 1]
    return len(hapax)


def calcolo_ttr(file_tokens):
    
    # Crea una lista vuota per contenere le TTR di ogni porzione di token
    ttrs = []
    # Calcola la TTR per ogni porzione di 200 token
    for i in range(200, len(file_tokens), 200):
        portion = ' '.join(file_tokens[i:i+200]) #Itero dall'elemento i a i+200 senza includere quest'ultimo,dato che i=200 inizialmente, prendo i primi 200, poi incremento di 200, se i starta da 0, mi calcola la ttr sui primi 0
        ttr = len(set(portion.split())) / len(portion.split()) #portion viene suddivisa in token, la converto in un set eliminando possibili duplicazioni, length di set = numero di token unici diviso il numero tot di token.
        ttr_limitato = "{:.3f}".format(ttr) #limito le cifre dec
        ttrs.append((i, ttr_limitato)) #alloco il risultato alla fine della lista, inizialmente è vuota e il primo valore prende posizione 0, poi posizione 1 per la seconda iterazione e cosi via.
        
    return ttrs


def pos_tagging(file):
    file = nltk.tag.pos_tag(file)
    return file    

def pos_tagging_FreqDist(k):
    file = k
    contatore = {tag: sum(1 for x, y in file if y == tag) for x, tag in file} # Mi restituisce un dizionario contenente la tag e il numero di occorrenze
    return contatore

def main(file1, file2):
    stdout_originale = sys.stdout
    file_output = open('outputprogrammaunotesto_uno_e_due.txt', 'w', encoding='utf-8')
    sys.stdout = file_output

    letturafile_uno = letturafile(file1)
    letturafile_due = letturafile(file2)
    
    # Tokenizzazione senza punteggiatura
    #tokenizer = RegexpTokenizer(r'\w+')
    #lettura_file_uno_reg_exp = tokenizer.tokenize(letturafile_uno)
    #lettura_file_due_reg_exp = tokenizer.tokenize(letturafile_due)
    

    # Lista tokens per file
    fileuno_tokens = get_tokens(letturafile_uno)
    filedue_tokens = get_tokens(letturafile_due)

    # Numero Tokens per file
    testouno_length = len(fileuno_tokens)
    testodue_length = len(filedue_tokens)
    print("\nIl file di testo uno ha:", testouno_length, "tokens")
    print("Il file di testo due ha:", testodue_length, "tokens")

    # Divido i corpus in frasi
    fileuno_frasi = nltk.tokenize.sent_tokenize(letturafile_uno)
    filedue_frasi = nltk.tokenize.sent_tokenize(letturafile_due)
    numerofrasi_fileuno = len(fileuno_frasi)
    numerofrasi_filedue = len(filedue_frasi)
    print("\nIl file di testo uno ha:", numerofrasi_fileuno, "frasi")
    print("Il file di testo due ha:", numerofrasi_filedue, "frasi")

    # Richiamo la funzione per calcolare la lunghezza media frasi per testo.
    Lunghezzamediafrasi_uno = LunghezzaMediaFrasi(fileuno_frasi, testouno_length)
    Lunghezzamediafrasi_due = LunghezzaMediaFrasi(filedue_frasi, testodue_length)
    print("\nLa lunghezza media delle frasi per il testo uno è:", Lunghezzamediafrasi_uno)
    print("La lunghezza media delle frasi per il testo due è:", Lunghezzamediafrasi_due)

    # Richiamo la funzione per calcolare la lunghezza media dei token
    Lunghezzamediatoken_uno= LunghezzaMediaToken(fileuno_tokens, testouno_length)
    Lunghezzamediatoken_due = LunghezzaMediaToken(filedue_tokens, testodue_length)
    print("\nNel file di testo uno abbiamo una lunghezza media dei token di:", Lunghezzamediatoken_uno)
    print("Nel file di testo due abbiamo una lunghezza media dei token di:", Lunghezzamediatoken_due)

    # Calcolo Hapax tra i primi 500,1000,3000 e nell'intero Corpus del file di testo uno

    primi_500  =  fileuno_tokens[:500]
    primi_1000 = fileuno_tokens[:1000]
    primi_3000 = fileuno_tokens[:3000]

    hapax_500 = hapax_contatore(primi_500)
    hapax_1000 = hapax_contatore(primi_1000)
    hapax_3000 = hapax_contatore(primi_3000)
    hapax_totali = hapax_contatore(fileuno_tokens)

    print("\nPer i primi 500 tokens abbiamo nel testo uno: ",hapax_500,"hapax")
    print("Per i primi 1000 tokens abbiamo nel testo uno: ", hapax_1000, "hapax")
    print("Per i primi 3000 tokens abbiamo nel testo uno: ", hapax_3000, "hapax")
    print("In totale nel testo uno abbiamo: ", hapax_totali, "hapax")

    # Calcolo Hapax tra i primi 500,1000,3000 e nell'intero Corpus del file di testo due

    primi_500 = filedue_tokens[:500]
    primi_1000 = filedue_tokens[:1000]
    primi_3000 = filedue_tokens[:3000]

    hapax_500 = hapax_contatore(primi_500)
    hapax_1000 = hapax_contatore(primi_1000)
    hapax_3000 = hapax_contatore(primi_3000)
    hapax_totali = hapax_contatore(filedue_tokens)

    print("Per i primi 500 tokens abbiamo nel testo due: ", hapax_500, "hapax")
    print("Per i primi 1000 tokens abbiamo nel testo due: ", hapax_1000, "hapax")
    print("Per i primi 3000 tokens abbiamo nel testo due: ", hapax_3000, "hapax")
    print("In totale nel testo due abbiamo: ", hapax_totali, "hapax")


    # Dimensione del vocabolario e ricchezza lessicale (Type-Token Ratio, TTR), calcolata
    # per porzioni incrementali di 200 token(i.e., i primi 200, i primi 400, i primi 600, …)
    
    ttr_fileuno = calcolo_ttr(fileuno_tokens)
    ttr_filedue = calcolo_ttr(filedue_tokens)
    
    print("\nTTR per il file uno con valore incrementale:", ttr_fileuno)
    print("TTR per il file due con valore incrementale:", ttr_filedue)

    # TTR totale nel testo uno
    types = len(set(fileuno_tokens))
    ttr = types/len(fileuno_tokens)
    print("\nTTR nel corpus uno:",ttr)

    # TTR totale nel testo due
    types = len(set(filedue_tokens))
    ttr = types/len(filedue_tokens)
    print("TTR nel corpus due:", ttr)



    # Calcolo numero di lemmi distinti
    # Utilizzerò WordNetLemmatizer() in quanto mi permette di lematizzare ogni token del corpus
    
    #Lo assegno ad una variabile per evitare errore in compilazione
    lematizzatore = WordNetLemmatizer()

    #lemmatizzo ogni singolo token del corpus
    lematizzatore_tokens_fileuno = [lematizzatore.lemmatize(x) for x in fileuno_tokens]
    lematizzatore_tokens_filedue = [lematizzatore.lemmatize(y) for y in filedue_tokens]

    #calcolo la frequenza dei lemmi
    lemma_freq_uno = FreqDist(lematizzatore_tokens_fileuno)
    lemma_freq_due = FreqDist(lematizzatore_tokens_filedue)
    
    #calcolo la dim del vocabolario dei lemmi.
    numero_lemmi_distinti_uno = len(set(lemma_freq_uno))
    numero_lemmi_distinti_due = len(set(lemma_freq_due))
    
    print("\nNel file uno abbiamo: ",numero_lemmi_distinti_uno,"di lemmi distinti")
    print("Nel file due abbiamo: ", numero_lemmi_distinti_due, "di lemmi distinti")


    token_pos_tag_fileuno = pos_tagging(fileuno_tokens)
    token_pos_tag_filedue = pos_tagging(filedue_tokens)

    Freq_Dist_Pos_uno = pos_tagging_FreqDist(token_pos_tag_fileuno)
    Freq_Dist_Pos_due = pos_tagging_FreqDist(token_pos_tag_filedue)

    
    #print(Freq_Dist_Pos_uno)
    #print(Freq_Dist_Pos_due)


    stdout_originale = sys.stdout
    file_output.close()

main(sys.argv[1], sys.argv[2])
