import pyttsx3
engine = pyttsx3.init("sapi5")  # Windows driver
engine.say("Hello Priti. Voice test successful.")
engine.runAndWait()
print("DONE")