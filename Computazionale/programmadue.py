import nltk
import sys
import string
import math
from collections import Counter
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import ConditionalFreqDist, ConditionalProbDist, MLEProbDist, FreqDist


def letturafile(file):
    with open(file, "r", encoding="utf-8") as file:
        testo = file.read()
    return testo

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
    return media


def pos_tagging(file):
    file = nltk.tag.pos_tag(file)
    return file


def pos_tagging_FreqDist(k):
    file = k
    # Mi restituisce un dizionario contenente la tag e il numero di occorrenze
    contatore = {tag: sum(1 for x, y in file if y == tag) for x, tag in file}
    return contatore

def primopunto(x,testo,k):
    tokens = nltk.word_tokenize(testo)
    pos_tags = nltk.pos_tag(tokens)
    Freq_Dist_Pos_Ordinate = x
    Freq_Dist_Pos_Ordinate = sorted(Freq_Dist_Pos_Ordinate.items(), reverse=True, key=lambda x: x[1])
    # Con key applico una func personalizzata per l'ordinamento, ovvero lambda che prende il secondo elemento di ogni tupla
    # tramite x[1], con reverse invece scelgo in che modo debba essere ordinata la funct sorted, items invece mi permettere
    # di ottenere una vista del dizionario con coppia-valore.
    # print(Freq_Dist_Pos_Ordinate)
    pos_freq_10 = Freq_Dist_Pos_Ordinate[:10]

    if(k==2):
        ngrammi = [(pos_tags[i][1], pos_tags[i+1][1]) for i in range(len(pos_tags)-1)]
        
    elif(k==3):
        ngrammi = [(pos_tags[i][1], pos_tags[i+1][1], pos_tags[i+2][1]) for i in range(len(pos_tags)-2)]
    ngrammi_counts = Counter(ngrammi)

    print("\nLista dei primi 10pos più frequenti: ", pos_freq_10)
    return(ngrammi_counts.most_common(10))


def primopuntopartedue(x, y):
    pos_tag = x
    filtro_pos = y
    lista_filtrata = [(token, pos) for (token, pos) in x if pos in filtro_pos]
    contatore = Counter(lista_filtrata)
    return contatore.most_common(20)

def secondopunto(x):
    pos_tag = x
    Aggettivi_Sostantivi = []
    for i in range(1, len(pos_tag)):
        if pos_tag[i-1][1].startswith('JJ') and pos_tag[i][1].startswith('NN'):
            Aggettivi_Sostantivi.append((pos_tag[i-1][0], pos_tag[i][0]))
    freq_distribuzione = nltk.FreqDist(Aggettivi_Sostantivi) #Calcolo la freq di ogni bigramma
    #print(Aggettivi_Sostantivi)
    freq_distrubuzione_venti =  freq_distribuzione.most_common(20) #seleziono i 20 piu' frequenti


    # I 20 con probabilità condizionata massima e relativo valore di probabilità.
    # Calcolo la distribuzione di frequenza condizionata
    cond_freq_dist = ConditionalFreqDist(Aggettivi_Sostantivi)

    # I 20 bigrammi con probabilità condizionata massima
    condizionata_massima = []       #itero su ogni aggettivo e per ciascuno determino il sostantivo che ha freq.max e calcolo relativa probabilità
    for aggettivo in cond_freq_dist.conditions(): #return elenco aggettivi unici presenti nella distribuzione di freq-condizionata.
        max_sostantivo = cond_freq_dist[aggettivo].max() #return il sostantivo con la frequenza massima per l'aggettivo specificato.
        max_prob = cond_freq_dist[aggettivo].freq(max_sostantivo) #calcola la probabilità condizionata dell'aggettivo dato il sostantivo con freq. massima.
        condizionata_massima.append((aggettivo, max_sostantivo, max_prob)) #li addo alla lista che contiene agg,sostantivo e relativa freq.max

    condizionata_massima = sorted(condizionata_massima, key=lambda x: x[2], reverse=True)[:20] #li ordino prendendo in considerazione la freq-max
    print("\nI 20 bigrammi con probabilità condizionata massima:")
    for bigramma, sostantivo, prob in condizionata_massima:
        print(bigramma, sostantivo, " - Probabilità:", prob)


    # I 20 con forza associativa (Pointwise Mutual Information, PMI) massima, e relativa PMI
    freq_bigrammi = Counter(Aggettivi_Sostantivi) #Calcolo la freq. dei bigrammi
    freq_totale = sum(freq_bigrammi.values()) #Calcolo la freq. totale dei lista contente i bigrammi
    PMI_bigrammi = {}
    frequenze_aggettivi = Counter([bigramma[0]for bigramma in freq_bigrammi]) #rappresente la frequenza degli aggettivi presenti nei bigrammi
    frequenze_sostantivi = Counter([bigramma[1]for bigramma in freq_bigrammi])  #rappresente la frequenza dei sostantivi presenti nei bigrammi

    for bigramma, frequenza in freq_bigrammi.items():
        aggettivo, sostantivo = bigramma

        frequenza_aggettivo = frequenze_aggettivi[aggettivo]
        frequenza_sostantivo = frequenze_sostantivi[sostantivo]

    # Calcola la PMI
        pmi = math.log2((frequenza/freq_totale)/((frequenza_aggettivo/freq_totale) * (frequenza_sostantivo / freq_totale)))
        PMI_bigrammi[bigramma] = pmi #dizionario che memorizza la pmi calcolata su ogni bigramma

    #Utilizzo lambda per riordinare
    bigrammi_pmi_ordinati = sorted(PMI_bigrammi.items(), key=lambda x: x[1], reverse=True)

    for bigramma, pmi in bigrammi_pmi_ordinati[:20]:
        print(f"Bigramma: {bigramma} - PMI: {pmi}")

    return freq_distrubuzione_venti
    




def terzopunto(x):
    testo = nltk.tokenize.sent_tokenize(x)
    frasi_accettate = []
    #una volta passato il testo filtro le frasi con len>=10 e <=20
    for frase in testo:
        if len(frase)>=10 and len(frase)<=20:
            frasi_accettate.append(frase)
    #print(frasi_accettate)

    frasi_filtrate = []
    for frase in frasi_accettate:
        frequenza = Counter(frase)
        #print(frase)
        #print(frequenza)
        contatore_hapax = sum(1 for x in frequenza if frequenza[x] == 1)
        if contatore_hapax <= len(frequenza)/2:
            frasi_filtrate.append((frase,frequenza))
    #print(frasi_filtrate)
    #calcolo media delle frequenze dei token in ogni frase e ritorno quella con la media maggiore.
    frase_media_maggiore = max(frasi_filtrate, key=lambda x: sum(testo.count(t) for t in x[0]) / len(x[0])) #for t in x[0] itera su ogni token nella frase x[0], sum(....) è una sum del conteggio di frequenza di ogni token nella frase, len(x[0]) è la length della frase formata dal numero di token, sum nel complesso fa la media della distr. di freq. dei token nella frase dividendo per la length della frase.
    frase_media_minore = min(frasi_filtrate, key=lambda x: sum(testo.count(t) for t in x[0]) / len(x[0]))
    print("\nLa frase con la media della distribuzione di frequenza dei token più alta è :",frase_media_maggiore[0])
    print("La frase con la media della distribuzione di frequenza dei token più bassa è :",frase_media_minore[0])
    return frasi_filtrate, frasi_accettate


def terzopuntotest(x):
    frasi_accettate = []
    token_corpus = []
    
    # Filtra le frasi con una lunghezza compresa tra 10 e 20 token
    for frase in nltk.tokenize.sent_tokenize(x):
        tokenized_frase = nltk.word_tokenize(frase)
        if len(tokenized_frase) >= 10 and len(tokenized_frase) <= 20:
            frasi_accettate.append(frase)
            token_corpus.extend(tokenized_frase)
    
    frasi_filtrate = []
    corpus_frequenza = Counter(token_corpus)
    
    # Filtra le frasi in cui almeno la metà dei token non è un hapax nell'intero corpus
    for frase in frasi_accettate:
        token_frase = nltk.word_tokenize(frase)
        frequenza = Counter(token_frase)
        contatore_hapax = sum(1 for token in token_frase if corpus_frequenza[token] >= 2)
        
        if contatore_hapax >= len(frequenza) / 2:
            frasi_filtrate.append((frase, frequenza))
    
    if len(frasi_filtrate) == 0:
        return None, None
    
    # Calcola la media delle frequenze dei token in ogni frase e trova quella con la media maggiore e minore
    frase_media_maggiore = max(frasi_filtrate, key=lambda x: sum(x[1][t] for t in x[1]) / len(x[1]))
    frase_media_minore = min(frasi_filtrate, key=lambda x: sum(x[1][t] for t in x[1]) / len(x[1]))
    print("\nFrase con media della distribuzione di freq più alta: ",frase_media_maggiore[0])
    print("Frase con media della distribuzione di freq più bassa: ",frase_media_minore[0])
    


def conditional_probability(trigram, trigram_freq_dist, bigram_freq_dist, tokens_freq_dist, corpus_size):

    if bigram_freq_dist[(trigram[0], trigram[1])] == 0:
        conditional_prob = 0
    else:
        conditional_prob = trigram_freq_dist[trigram] / \
            bigram_freq_dist[(trigram[0], trigram[1])]

    return conditional_prob



def markov2(sentence, trigram_freq_dist, bigram_freq_dist, tokens_freq_dist, corpus_size):
    tokens = nltk.tokenize.word_tokenize(sentence)
    trigrams = list(nltk.trigrams(tokens))

    # Calcola la probabilità del primo token nel corpus
    bigram = (tokens[0], tokens[1])
    if tokens_freq_dist[bigram[0]] == 0:
        prob = 0
    else:
        prob = bigram_freq_dist[bigram]*1.0/tokens_freq_dist[bigram[0]]*1.0

    # Calcola la probabilità per ogni trigramma nella frase
    trigram = None
    for trigram in trigrams:
            cond_prob = conditional_probability(trigram, trigram_freq_dist, bigram_freq_dist, tokens_freq_dist, corpus_size)
            prob = prob*cond_prob
            
    return prob,trigram




def quartopunto(x):
    pos_tag = x
    #estraggo le EN dal testo
    chunk_testo = nltk.ne_chunk(pos_tag)
    nomi = []
    luoghi = []
    persone = []
    for y in chunk_testo:
        #controllo che l'oggetto ha l'attributo specifico hasattr mi permette di fare ciò
        if hasattr(y, 'label'): 
            if y.label() == 'PERSON':
                persone.append(''.join([c[0] for c in y]))
            elif y.label() == 'GPE':
                luoghi.append(''.join([c[0] for c in y]))
            elif y.label() == 'ORGANIZATION':
                nomi.append(''.join(c[0] for c in y)) #creazione di una stringa con concatenazione dei token delle entità riconosciuta come organization, c[0] prendo il primo elem di ogni tupla c dentro y.
    
    contatore_persone = Counter(persone)
    contatore_luoghi = Counter(luoghi)
    contatore_nomi = Counter(nomi)

    ordinamento_persone_decrescente = contatore_persone.most_common(15)
    print("\nOrdinamento NE persone decrescente: ", ordinamento_persone_decrescente) 
    ordinamento_luoghi_descrescente = contatore_luoghi.most_common(15)
    print("\nOrdinamento NE luoghi decrescente: ",ordinamento_luoghi_descrescente)
    ordinamento_nomi_decrescente = contatore_nomi.most_common(15)
    print("\nOrdinamento NE nomi decrescente: ", ordinamento_nomi_decrescente)

    return ordinamento_persone_decrescente,ordinamento_luoghi_descrescente,ordinamento_nomi_decrescente



def main(file1):
    stdout_originale = sys.stdout
    file_output = open('output_programmadue_filedue.txt', 'w', encoding='utf-8')
    sys.stdout = file_output

    testo = letturafile(file1)
    
    #Lista tokens nel file
    testo_tokens = get_tokens(testo)
    
    
    #Numero di frasi
    testo_numerofrasi = nltk.tokenize.sent_tokenize(testo)
    numerofrasi_fileuno = len(testo_numerofrasi)
    print("\nIl file di testo ha:", numerofrasi_fileuno, "frasi")

    #Pos Tagging
    pos_tag = pos_tagging(testo_tokens)
    #print(pos_tag)

    Freq_Dist_Pos = pos_tagging_FreqDist(pos_tag)
    #print(Freq_Dist_Pos)
    
    bigrammi = primopunto(Freq_Dist_Pos, testo,  2)
    trigrammi = primopunto(Freq_Dist_Pos, testo, 3)
    print("\nLista bigrammi top 10 : ", bigrammi)
    print("Lista trigrammi top 10 :", trigrammi )

    sostantivi = ['NN', 'NNS', 'NNP', 'NNPS']
    aggettivi = ['JJ', 'JJR', 'JJS']
    avverbi = ['RB', 'RBR', 'RBS']

    print("\nI 20 sostantivi più frequenti: ", primopuntopartedue(pos_tag,sostantivi))
    print("I 20 aggettivi più frequenti: ", primopuntopartedue(pos_tag,aggettivi))
    print("I 20 avverbi più frequenti: ", primopuntopartedue(pos_tag,avverbi))

    puntodue = secondopunto(pos_tag)
    print("\nLista dei 20 bigrammi aggettivi,sostantivi più frequenti: ",puntodue)

    
    terzopuntotest(testo)
    
    quartopunto(pos_tag)
    
    test_sentences = nltk.tokenize.sent_tokenize(testo)
    corpus_tokens = nltk.tokenize.word_tokenize(testo)
    tokens_freq_dist = nltk.FreqDist(corpus_tokens)
    bigrams = list(nltk.bigrams(corpus_tokens))
    bigrams_freq_dist = nltk.FreqDist(bigrams)
    trigrams = list(nltk.trigrams(corpus_tokens))
    trigrams_freq_dist = nltk.FreqDist(trigrams)

    # Probabilità frase sull'intero corpus
    lista = []
    for sentence in test_sentences:
        # print(sentence)
        probabilità, trigramma = markov2(sentence, trigrams_freq_dist, bigrams_freq_dist, tokens_freq_dist, len(corpus_tokens))
        lista.append((probabilità, trigramma))
    lista_ordinata = sorted(lista, key=lambda x: x[0], reverse=True)
    max_valore = max(lista, key=lambda x: x[0])
    print("\nLa frase con probabilità maggiore nell'intero corpus,secondo un modello di Markov di ordine 2, è : ", max_valore)



    stdout_originale = sys.stdout
    file_output.close()

main(sys.argv[1])
