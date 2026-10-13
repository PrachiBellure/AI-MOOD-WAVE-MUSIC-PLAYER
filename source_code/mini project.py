import cv2
from deepface import DeepFace
import pandas as pd
import webbrowser
import time

# -------------------------
# Full song dataset (exactly as provided)
# -------------------------
songs = [
    # HAPPY SONGS
    {"emotion": "happy", "language": "hindi", "song_name": "O meri Zhorajabeen", "artist":"Himesh Reshammiya", "youtube_link": "https://youtu.be/9FPAdxhYZIc?si=idEa_IMzldVA9amf"},
    {"emotion": "happy", "language": "hindi", "song_name":"Bhootni Ke", "artist":"Daler Mehndi", "youtube_link": "https://youtu.be/pNLW5d02BOA?si=LXpRiK3mFqtJdIs9"},
    {"emotion": "happy", "language": "hindi", "song_name": "Lets nacho", "artist":"Badshah", "youtube_link": "https://youtu.be/TLnwqAarPkM?si=wMsHfsDspdWIGdsZ"},
    {"emotion": "happy", "language": "hindi", "song_name": "Badtameez Dil", "artist":"Benny Dayal", "youtube_link": "https://youtu.be/II2EO3Nw4m0?si=lHtrCjKV46JuXUQ3"},
    {"emotion": "happy", "language": "hindi", "song_name": "Saturday saturday", "artist":"sharib toshi", "youtube_link": "https://youtu.be/rW9_-dVCmrM?si=rCCrUt_Hyud4PdcB"},

    {"emotion": "happy", "language": "english", "song_name": "Bob Marley crying laf", "artist": "Bob Marley", "youtube_link": "https://youtu.be/LowiRNulPAE?si=Tk1Rc5ijAQnkoKR5"},
    {"emotion": "happy", "language": "english", "song_name": "die with a smile", "artist": "Lady Gaga,Bruno mars", "youtube_link": "https://www.youtube.com/watch?v=OPf0YbXqDm0"},
    {"emotion": "happy", "language": "english", "song_name": "believer", "artist": "imagine Dragons", "youtube_link": "https://youtu.be/W0DM5lcj6mw?si=FOqNXVTNF4fw0fyqN"},
    {"emotion": "happy", "language": "english", "song_name": "shutup and dance", "artist": "walk the moon", "youtube_link": "https://youtu.be/97_Bdpq7Myk?si=PxSD68YjWQzJ04tr"},
    {"emotion": "happy", "language": "english", "song_name": "levitating", "artist": "Dua lipa", "youtube_link": "https://youtu.be/ovG2QwiObhk?si=PnsjeaJXWa4U7XNd"},
    
    {"emotion": "happy", "language": "telugu", "song_name": "gunna gunna mamidi", "artist": "sai katheek", "youtube_link": "https://youtu.be/5lbLSfEdBeI?si=dXdQUn6PTradWCrU"},
    {"emotion": "happy", "language": "telugu", "song_name": "golden sparrow", "artist": "GV Prakash", "youtube_link": "https://youtu.be/1seR_ckLXz4?si=B9Bw7qaF57ka6JNQ"},
    {"emotion": "happy", "language": "telugu", "song_name": "saranga dariya", "artist": "Mangli", "youtube_link": "https://youtu.be/1ozmyl1ZEyY?si=hTo-bJSotCZeHX5F"},
    {"emotion": "happy", "language": "telugu", "song_name": "bullet", "artist":"Silambarasan TR & haripriya" ,"youtube_link": "https://youtu.be/WgrLE4Fqxeo?si=YzVnrurc7KLPnYAd"},   
    {"emotion": "happy", "language": "telugu", "song_name": "Ranjithame", "artist": " anurag kulkarni,M M Manasi", "youtube_link": "https://youtu.be/FjjYlp4pm7E?si=dYzxJ56Jg1hSzojE"},

    {"emotion": "happy", "language": "kannada", "song_name": "Belageddu", "artist": "Vijay Prakash", "youtube_link": "https://youtu.be/ebz20FHrT44?si=a4bfqULVJTK2MNaO"},
    {"emotion": "happy", "language": "kannada", "song_name": "ALL OK", "artist": " ALL OK", "youtube_link": "https://youtu.be/hRkc-OPHApY?si=a27bpSh4DuuVHw9L"},
    {"emotion": "happy", "language": "kannada", "song_name": "Jolly go jolly go", "artist": "Gurukiran", "youtube_link": "https://youtu.be/mFjiqEobngA?si=Cpe19s-5YodU-d31"},
    {"emotion": "happy", "language": "kannada", "song_name": "Lifuu istene", "artist": "chetan sosca", "youtube_link": "https://youtu.be/gCyW7lFXR_A?si=4Bf1WluxeZ1a5C7W"},
    {"emotion": "happy", "language": "kannada", "song_name": "Hodi ombath", "artist": "vijay prakash", "youtube_link": "https://youtu.be/7uFWC5xNnYQ?si=B--zIftEV3xsG-bE"},

    #SAD SONGS
    {"emotion": "sad","language": "hindi", "song_name": "illahi", "artist": "arjit singh", "youtube_link": "https://youtu.be/6w67NOaRe-w?si=vBgpoAqxMx4F7wrL"},
    {"emotion": "sad","language": "hindi", "song_name": "iktara", "artist": "kavita seth", "youtube_link": "https://youtu.be/akjdj6iHttY?si=h0LGbCLwnAPTLQGD"},
    {"emotion": "sad", "language": "hindi", "song_name": "all is well", "artist": "sonu nigam,swanand krikire,shaan", "youtube_link": "https://youtu.be/7PzwOiW8-n0?si=HP5An2pP6FYaDtPQ"},
    {"emotion": "sad", "language": "hindi", "song_name": "tere bina", "artist": "a.r.rahman", "youtube_link": "https://youtu.be/9JDSGhhiOwI?si=rmhx67XshQm_YvZU"},
    {"emotion": "sad","language": "hindi", "song_name": "phir se ud chala", "artist": "mohit chauhan", "youtube_link": "https://youtu.be/2mWaqsC3U7k?si=I7TBbaMh-IULu5_f"},

    {"emotion": "sad", "language": "english", "song_name": "Wham!", "artist": "GeorgeMichael", "youtube_link": "https://youtu.be/pIgZ7gMze7A?si=w5oE22Hk3urslHAT"},
    {"emotion": "sad", "language": "english", "song_name": "Katrina & the waves", "artist": "Katrina", "youtube_link": "https://youtu.be/iPUmE-tne5U?si=UGwf0CJhlrBbKvre"},
    {"emotion": "sad", "language": "english", "song_name": "believer", "artist": "imagine Dragons", "youtube_link": "https://youtu.be/W0DM5lcj6mw?si=FONXVTNF4fw0fyqN"},
    {"emotion": "sad", "language": "english", "song_name": "happy together", "artist": "turtle", "youtube_link": "https://youtu.be/9ZEURntrQOg?si=8cmd4bhWIA2i2yc5"},
    {"emotion": "sad", "language": "english", "song_name": "Barenaked ladies", "artist": "Dua lipa", "youtube_link": "https://youtu.be/fC_q9KPczAg?si=aESYLOFl42GFydMo"},

    {"emotion": "sad", "language": "telugu", "song_name": "egire mabbulalona", "artist": "yuvan shankar", "youtube_link": "https://youtu.be/iiBu2aGWlcg?si=I3-42L_Kl74F3XTP"},
    {"emotion": "sad", "language": "telugu", "song_name": "Aaradhya", "artist": "Hesham Abdul Wahab", "youtube_link": "https://youtu.be/MBlgaPguCaQ?si=_6aTQbSLJkoWEXuE"},
    {"emotion": "sad", "language": "telugu", "song_name": "samayama", "artist": "kiara khanna", "youtube_link": "https://youtu.be/Zz1M1iVEkwM?si=O_rNI6q50Ahqc3wN"},
    {"emotion": "sad", "language": "telugu", "song_name": "hey pillagaada", "artist": "sinduri", "youtube_link": "https://youtu.be/k9DMXBFHEH4?si=qGgx2GjHRisLwyAP"},
    {"emotion": "sad", "language": "telugu", "song_name": "pranavalaya", "artist": " Anurag kulkarni", "youtube_link": "https://youtu.be/4CKFAb1FNns?si=Pnku0T-Olqh6fHfk"},

    {"emotion": "sad","language": "kannada", "song_name": "akasha iste yakideyo","artist": "tippu,kunal ganjawala", "youtube_link": "https://youtu.be/HfNtBJSkprk?si=hroZuHomA2-GrYAW"},
    {"emotion": "sad","language": "kannada", "song_name": "bhuvanam gaganam", "artist": "ravi kale", "youtube_link": "https://youtu.be/NujNxgy4CpA?si=RVfLLR6M-eW_G9gg"},
    {"emotion": "sad", "language": "kannada", "song_name": "gagana nee", "artist": "suchetra", "youtube_link": "https://youtu.be/ywnfyiu7c_k?si=clq6fQsBwQ8gydQG"},
    {"emotion": "sad", "language": "kannada", "song_name": "jaago re jaago", "artist": "kunal ganjawala", "youtube_link": "https://youtu.be/2KHO1a7YfJQ?si=43HV_FnhinTC1QkR"},
    {"emotion": "sad","language": "kannada", "song_name": "jagave ninadu", "artist": "benny dayal,asif akbar", "youtube_link": "https://youtu.be/mH-c1ZdE2YM?si=p0ib4ynOvOPsK3GI"},

    # ANGRY SONGS
    {"emotion": "angry", "language": "hindi", "song_name": "kaun tujhe", "artist": "amaal malik palak", "youtube_link": "https://youtu.be/JHUrRSBtUNE?si=c1EFJxTb6a1Y8cqA"},
    {"emotion": "angry", "language": "hindi", "song_name": "agar tum sath ho", "artist": "alka yagnik,aruit singh", "youtube_link": "https://youtu.be/sK7riqg2mr4?si=i0zmI0lbQhi_0_Kj"},
    {"emotion": "angry", "language": "hindi", "song_name": "aavan jaavan ", "artist": "arjith sing", "youtube_link": "https://youtu.be/enjkcCdAlXc?si=-nADsMwRsGxyGCJo"},
    {"emotion": "angry", "language": "hindi", "song_name": "kal ho na ho", "artist": "sonu nigam", "youtube_link": "https://youtu.be/g0eO74UmRBs?si=BjxC6rsEn-FcUu6b"},
    {"emotion": "angry", "language": "hindi", "song_name": "adharam madhuram", "artist": "sohini mishra", "youtube_link": "https://youtu.be/tBgNpc39FJk?si=gPTAXOGI0ZyCgwtv"},

    {"emotion": "angry", "language": "telugu", "song_name": "the soul of radhe shyam", "artist": "justin prabhakaran", "youtube_link": "https://youtu.be/DlEIVZE0YC0?si=nPev5ratTIYsLriz"},
    {"emotion": "angry", "language": "telugu", "song_name": "puyhiyoru lokam", "artist": "vimal roy", "youtube_link": "https://youtu.be/lZL2K3ewRYM?si=h-K-oN-S7f-8x3OT"},
    {"emotion": "angry", "language": "telugu", "song_name": "nuvenna ", "artist": "shreya ghoshal", "youtube_link": "https://youtu.be/vY8caUaAUUM?si=uKFfsZiLdog6WRLE"},
    {"emotion": "angry", "language": "telugu", "song_name": "oohalu gusagusalade", "artist": "deepu,sravani", "youtube_link": "https://youtu.be/DDb7OILQMMA?si=Xnc_Q5SarxlrPh7O"},
    {"emotion": "angry", "language": "telugu", "song_name": "achyutam keshavam ", "artist": "alka yagnik", "youtube_link": "https://youtu.be/5-Xoh7jKVo8?si=sJhsvLzeFIZoTzK8"},

    {"emotion": "angry", "language": "english", "song_name": "dandelions", "artist": "ruth b", "youtube_link": "https://youtu.be/WgTMeICssXY?si=y_jL9rhQ9pVGjVXM"},
    {"emotion": "angry", "language": "english", "song_name": "memories", "artist": "maroon 5", "youtube_link": "https://youtu.be/o2DXt11SMNI?si=0e7ROxMsdIz952r4"},
    {"emotion": "angry", "language": "english", "song_name": "stay ", "artist": "justin bieber", "youtube_link": "https://youtu.be/yWHrYNP6j4k?si=6djXEWjbJgGFiVdu"},
    {"emotion": "angry", "language": "english", "song_name": "Co2", "artist": "preteek kuhad", "youtube_link": "https://youtu.be/rrCYMsV7A-c?si=-G3de401FBU96ePm" },
    {"emotion": "angry", "language": "english", "song_name": "those eyes ", "artist": "new west", "youtube_link": "https://youtu.be/t1dvrcqlQgI?si=q1Wne72a_qB440qm"},

    {"emotion": "angry", "language": "kannada", "song_name": "olave olave", "artist": "srilakshmi belmannu", "youtube_link": "https://youtu.be/JUNsMqN6bb0?si=X2_PtbFRSd2zlXgs"},
    {"emotion": "angry", "language": "kannada", "song_name": "ade bhoomi ade bhanu", "artist": "song nigum", "youtube_link": "https://youtu.be/CVnfvVGnspU?si=ghDsMt5flAAfi9nw"},
    {"emotion": "angry", "language": "kannada", "song_name": " kaagadada doniyalli", "artist": "vasuki vaibhav", "youtube_link": "https://youtu.be/EtGh9oC2SZ0?si=YgEd1YpSV9B2fGw3"},
    {"emotion": "angry", "language": "kannada", "song_name": "mukunda murari", "artist": "shankar mahadevan", "youtube_link": "https://youtu.be/LKu26KlYE7s?si=2JYObRZz7hSqRkvf" },
    {"emotion": "angry", "language": "kannada", "song_name": " brahmakalasha", "artist": "abby v", "youtube_link": "https://youtu.be/palMj0iq-3g?si=QpP_FRsosBDAhM2J"},
    
    # NEUTRAL SONGS
    {"emotion": "neutral", "language": "hindi", "song_name": "matargashti", "artist": "mohit chauhan", "youtube_link": "https://youtu.be/6vKucgAeF_Q?si=YJGloHq4HnQ0kZ8h"},
    {"emotion": "neutral", "language": "hindi", "song_name": "apna har din aise jiyo", "artist": "shaan,anouska manchanda", "youtube_link": "https://youtu.be/tKmDt5H5Z9w?si=26kyUffI8mhIn96L"},
    {"emotion": "neutral", "language": "hindi", "song_name": "love you zindagi", "artist": "amit trivedi", "youtube_link": "https://youtu.be/bw7bVpI5VcM?si=IIzh5bI-fbd5HEnT"},
    {"emotion": "neutral", "language": "hindi", "song_name": "gallan goodiyaan", "artist": "yashita sharma,manish", "youtube_link": "https://youtu.be/jCEdTq3j-0U?si=Dou9OG1eZUBPyFxJ" },
    {"emotion": "neutral", "language": "hindi", "song_name": "kashmir main tu kanyakumari", "artist": "sunidhi chauhan,arjith singh,neeti mohan", "youtube_link": "https://youtu.be/WxtJqyIyThU?si=o4WC5kjHYZCGgaxn"},

    {"emotion": "neutral", "language": "english", "song_name": "that's so true", "artist": "gracie abrams", "youtube_link": "https://youtu.be/G_lNBrDJQC8?si=qpUzIlJpWhDjVNcv"},
    {"emotion": "neutral", "language": "english", "song_name": "expresso", "artist": "sabina carpenter", "youtube_link": "https://youtu.be/4h0a1NCeC14?si=nFpQA6dZb74WMqxq"},
    {"emotion": "neutral", "language": "english", "song_name": "i like me better", "artist": "lauv", "youtube_link": "https://youtu.be/a7fzkqLozwA?si=O47GOQu-9qXgRrNU"},
    {"emotion": "neutral", "language": "english", "song_name": "intensions", "artist": "justin bieber", "youtube_link": "https://youtu.be/3AyMjyHu1bA?si=tOcmNPXsgzOgtisD" },
    {"emotion": "neutral", "language": "english", "song_name": "night changes", "artist": "liam,zyan,harry,louis", "youtube_link": "https://youtu.be/bMBdqvJWofQ?si=5c2x_oaAfuUGTOkb"},

    {"emotion": "neutral", "language": "telugu", "song_name": "janatha garage", "artist": "devi shri prasad", "youtube_link": "https://youtu.be/Fa4COn3sPDY?si=xFnDq9a_SbdtKqCu"},
    {"emotion": "neutral", "language": "telugu", "song_name": "crazy feeling", "artist": "prudhvi chandra", "youtube_link": "https://youtu.be/QCTtc36u-Kk?si=3djEIvAwZP5GRLoo"},
    {"emotion": "neutral", "language": "telugu", "song_name": "appudo ippudo", "artist": "siddharth", "youtube_link": "https://youtu.be/X_HleAX9jVM?si=wQyEke59XbDZ7r-o"},
    {"emotion": "neutral", "language": "telugu", "song_name": "kalaavathi", "artist": "sid sriram", "youtube_link": "https://youtu.be/lcfcTbEppQM?si=HKS1WHyZiusnFzbE" },
    {"emotion": "neutral", "language": "telugu", "song_name": "chiru chiru", "artist": "hari charan", "youtube_link": "https://youtu.be/hCt-H4-5wco?si=npAIOfzVzLhcGGCf"},

    {"emotion": "neutral", "language": "kannada", "song_name": "danks anthem", "artist": "anurag kulkarni", "youtube_link": "https://youtu.be/RldAVzPGMuA?si=qh5OUAdy0DYVEHPz"},
    {"emotion": "neutral", "language": "kannada", "song_name": "munajane majalli", "artist": "raghu dixit,lakshmi manmohan", "youtube_link": "https://youtu.be/xNR4FAEGxV4?si=xJDnvzlPF3WYpDV7"},
    {"emotion": "neutral", "language": "kannada", "song_name": "yeno yeno aagide", "artist": "haricharan & apporva", "youtube_link": "https://youtu.be/xNR4FAEGxV4?si=xJDnvzlPF3WYpDV7"},
    {"emotion": "neutral", "language": "kannada", "song_name": "dwapara", "artist": "jaskaran singh", "youtube_link": "https://youtu.be/wiur_AGatGU?si=TS4MuQxna7jOrzBl" },
    {"emotion": "neutral", "language": "kannada", "song_name": "namaami namaami", "artist": "aishwarya rangarajan", "youtube_link": "https://youtu.be/StNb3Jbwm6o?si=bgRSUP5JLK51Qrvp"},
   
]

df = pd.DataFrame(songs)

# -------------------------
# Step 1: Open webcam & capture frame when 'q' is pressed
# -------------------------
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("🎥 Webcam started. Show your emotion and press 'q' (lowercase) to capture.")
print("Press 'ESC' to exit without capturing.")

captured_frame = None

while True:
    ret, frame = cam.read()
    if not ret:
        print("Error accessing camera.")
        break

    # Optional: Resize preview for performance (uncomment if needed)
    # frame_small = cv2.resize(frame, (640, 480))
    cv2.imshow("Press 'q' to Capture Emotion (or ESC to quit)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # capture on 'q'
        captured_frame = frame.copy()
        print("📸 Image captured!")
        break
    elif key == 27:  # ESC key to quit
        print("Exiting without capture.")
        break

cam.release()
cv2.destroyAllWindows()

# -------------------------
# Step 2: Analyze captured frame
# -------------------------
if captured_frame is None:
    exit()

try:
    result = DeepFace.analyze(captured_frame, actions=['emotion'], enforce_detection=False)
    # DeepFace.analyze sometimes returns a dict or a list containing dict
    if isinstance(result, list):
        dominant_emotion = result[0].get('dominant_emotion', '').lower()
    else:
        dominant_emotion = result.get('dominant_emotion', '').lower()
except Exception as e:
    print("Error analyzing emotion:", e)
    exit()

if not dominant_emotion:
    print("Could not detect dominant emotion. Exiting.")
    exit()

print(f"\nDetected Emotion: {dominant_emotion.capitalize()}")

# -------------------------
# Step 3: Ask for language choice
# -------------------------
language = input("\nEnter preferred language (hindi / english / telugu / kannada): ").strip().lower()

# -------------------------
# Step 4: Filter songs and display
# -------------------------
filtered_songs = df[(df['emotion'] == dominant_emotion) & (df['language'] == language)].reset_index(drop=True)

if filtered_songs.empty:
    print("\nNo songs available for this emotion and language.")
    exit()

print(f"\n🎵 Songs for your emotion '{dominant_emotion.capitalize()}' in {language.title()}:\n")
for idx, row in filtered_songs.iterrows():
    print(f"{idx+1}. {row['song_name']} - {row['artist']}")
    print(f"   YouTube: {row['youtube_link']}\n")

# -------------------------
# Step 5: Play selected song (safe input handling)
# -------------------------
while True:
    try:
        choice = int(input("Enter the number of the song you want to play: ")) - 1
        if 0 <= choice < len(filtered_songs):
            selected_song = filtered_songs.iloc[choice]
            print(f"\nPlaying: {selected_song['song_name']} 🎶")
            webbrowser.open(selected_song['youtube_link'])
            break
        else:
            print(f"Please enter a number between 1 and {len(filtered_songs)}.")
    except ValueError:
        print("Please enter a valid number.")
