import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# put the file name as the sourceFile variable that contains the input text in this format:
    # 0,lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua
    # 1,ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur
    # 0,excepteur sint occaecat cupidatat non proident sunt in culpa qui officia deserunt mollit anim id est laborum

    # a pos-tag version might look similar to this:
    # 0,cd fw nn prp vbd dt nn in jj vbd in dt nn wp vbz prp
    # 1,cd nnp vbz dt nn rp
    # 0,cd fw nn dt vbz dt jj nn nn nn nn vbd rb vbn in dt nn nn rb vbz to prp$ nn in vbg to vb dt jj nn dt jj nn md vb to vb prp rp wrb md dt nn nn

sourceFile = "/Users/clifforeilly/Documents/Projects/PhD/Lariat3/lariat4/litotes-tweets-pos-v1.csv"
df = pd.read_csv(sourceFile)

training_data = pd.DataFrame()
testing_data = pd.DataFrame()
training_data, testing_data = train_test_split(df, test_size=0.2) # no random_state parameter means this is randomly split each time


# calculate sizes and print out the distribution of the training data
train_size = 0
train_lit = 0
for index, x in training_data.iterrows():
    train_size += 1
    tg = x['0']
    if tg == 1:
        train_lit += 1

print("training set size=" + str(train_size))
print("training set Litotes count=" + str(train_lit))
train_split = train_lit / train_size
print("training set split=" + str(train_split))

# calculate sizes and print out the distribution of the testing data
test_size = 0
test_lit = 0
for index, x in testing_data.iterrows():
    test_size += 1
    tg = x['0']
    if tg == 1:
        test_lit += 1

print("testing set size=" + str(test_size))
print("testing set Litotes count=" + str(test_lit))
test_split = test_lit / test_size
print("testing set split=" + str(test_split))


# process input file into the correct shapes
xtr = []
ytr = []
for index, x in training_data.iterrows():
    ytr.append(x['0'])
    xtr.append(str(x['1']))

xte = []
for index, x in testing_data.iterrows():
    xte.append(str(x['1']))

vec = CountVectorizer()
x_train = vec.fit_transform(xtr)
x_test = vec.transform(xte)

# vary these model parametsr to change the algorithm, e.g. penalty='l1', C=0.1
logreg = LogisticRegression(penalty="l1",
    solver="liblinear",
    tol=1e-6,
    max_iter=int(1e6),
    warm_start=True,
    intercept_scaling=10000,
    C=0.5)

scaler = StandardScaler(with_mean=False).fit(x_train)
XS = scaler.transform(x_train)
logreg.fit(XS, ytr)
lr_prediction = logreg.predict(x_test)
lr_score = logreg.score(XS, ytr)

print("model score=" + str(lr_score))
print("finished")
