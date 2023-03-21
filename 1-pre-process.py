import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import ssl
import pandas as pd

print("lariat4 started ...")

# these lines prevent an error with downloading the nltk resources
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

nltk.download('words')
nltk.download('omw-1.4')
dictn = set(nltk.corpus.words.words())

# uncomment this section to help remove stop words (further process would be needed in the code to do the removal)
#nltk.download('stopwords')
#stop_words = stopwords.words('english')

print("imported nltk...")

litotesCues = {"not uncommon", "not unhappy", "not unpleasant", "not  unhappy", "not  unpleasant", "not  uncommon", "not uncommon", "not  unaware", "not  unfamiliar", "not unaware", "not unfamiliar"}
print("litotes cue phrases: " + str(litotesCues))

lemmatizer = WordNetLemmatizer()
process = "Pre-Processing"
print("started " + process)

# put the file name as the sourceFile variable that contains the input text in this format:
    # 0,Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
    # 1,Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. 
    # 0,Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

sourceFile = ""
print("reading file " + sourceFile + "...")
file1 = open(sourceFile, 'r')
Lines = file1.readlines()

print("starting input source file and processing... ")

# comment out the value for type depending on whether the input text should contain multiple sentences (e.g. potentially a 'tweet') or a single sentence (i.e. break up the input into discrete sentence forms)
#type = 'single'
type = 'multi'

sentences = [] 
targets = []
cntw = 0
cnts = 0
cntm = 0
separator = " "

for line in Lines:
    target = line.split(",")[0]
    targets.append(int(target))
    text = line

    # validate whether the text actually does contain one of the cue phrase in the litotesCues array and mark positive if so
    genuinelitotes = False
    for c in litotesCues:
        if c in text:
            genuinelitotes = True
            
    # correct the target if the input text does actually contain the litotes cue phrase
    if genuinelitotes is False and target == '1':
        cntw += 1
        target = '0'

    if genuinelitotes is True and target == '0':
        cntw += 1
        target = '1'

    words = []

    if type == "multi":
        tokens = word_tokenize(text)
        tokens = [w.lower() for w in tokens]
        tokens = [lemmatizer.lemmatize(w) for w in tokens]
        words = [word for word in tokens if word.isalpha() and word in dictn] # remove 'if word.isalpha() and word in dictn' to broaden the output to beyond just English words
        tagz = []
        #Convert tokens into pos tags
        tokep = nltk.pos_tag(words)
        for ws in tokep:
            tag = ws[1]
            if tag != None:
                tagz.append(tag.lower())
        
        w2 = separator.join(tagz)
    
        if len(tagz) > 1:
            xw = [target, w2]
            sentences.append(xw)
            cntm += 1

    if type == "single":
        sents = sent_tokenize(text)
        for s in sents:
            tokens = word_tokenize(s)
            tokens = [w.lower() for w in tokens]
            tokens = [lemmatizer.lemmatize(w) for w in tokens]
            words = [word for word in tokens if word.isalpha() and word in dictn] # remove 'if word.isalpha() and word in dictn' to broaden the output to beyond just English words
            tagz = []
            #Convert tokens into pos tags
            tokep = nltk.pos_tag(words)
            for ws in tokep:
                tag = ws[1]
                if tag != None:
                    tagz.append(tag.lower())
            
            w2 = separator.join(tagz)
        
            if len(tagz) > 1:
                xw = [target, w2]
                sentences.append(xw)
                cnts += 1

print("sentence count = " + str(cnts))
print("multi sentence (tweet) count = " + str(cntm))
print("wrong target count = " + str(cntw))

#remove duplicates
seen = set()
result = []
cnt = 0
for item in sentences:
    cnt += 1
    t = item[0]
    x = item[1]
    if x not in seen:
        seen.add(x)
        result.append(item)


df = pd.DataFrame(result)

# add the file name to save as (.csv) into the saveToFilename variable
saveToFilename = ""
df.to_csv(saveToFilename, index = False, index_label= False)

print("finished " + process)
