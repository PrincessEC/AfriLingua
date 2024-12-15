# This program translates common English phrases to other languages
# Import required libraries
import json
import os

# function to load phrases from the Json file
def loadPhrases():
    try:
        # Check if Json file exists, if not create empty Json
        if os.path.exists('translations.json'):
            with open('translations.json', 'r') as file:
                return json.load(file)
        else:
            # Adding default phrases
            defaultPhrases = {
                "How are you?": {"Spanish": "¿Cómo estás?", "French": "Comment allez-vous?"},
                "Good morning": {"Spanish": "Buenos días", "French": "Bonjour"},
                "Thank you": {"Spanish": "Gracias", "French": "Merci"},
                "You're welcome": {"Spanish": "De nada", "French": "De rien"},
                "Goodbye": {"Spanish": "Adiós", "French": "Au revoir"},
                "Please": {"Spanish": "Por favor", "French": "S'il vous plaît"},
                "Nice to meet you": {"Spanish": "Encantado/a", "French": "Enchanté"},
                "What's your name?": {"Spanish": "¿Cómo te llamas?", "French": "Comment tu t'appelles?"},
                "I don't understand": {"Spanish": "No entiendo", "French": "Je ne comprends pas"},
                "Have a nice day": {"Spanish": "Que tengas un buen día", "French": "Bonne journée"}
            }
            savePhrases(defaultPhrases)
            return defaultPhrases
    except:
        print("Error loading phrases!")
        return {}

# Function to save phrases to json file
def savePhrases(phrases):
    try:
        with open('translations.json', 'w') as file:
            json.dump(phrases, file)
    except:
        print("Error saving phrases!")





# Function to display main menu
def showMenu():
    print("\n=== Phrase Translator ===")
    print("1. View all phrases")
    print("2. Add new phrase")
    print("3. Translate a phrase")
    print("4. Exit")
    print("=====================")



# Main program
def main():
    phrases = loadPhrases()
    
    while True:
        showMenu()
        
        # Get user menu choice
        choice = input("\nChose an option (1-4): ")
        
        if choice == '1':
            # View all phrases
            print("\nAvailable Phrases:")
            for phrase in phrases:
                print(f"\nEnglish: {phrase}")
                for language, translation in phrases[phrase].items():
                    print(f"{language}: {translation}")
        
        
        elif choice == '2':
            # Add new phrase
            english = input("\nEnter English phrase: ")
            language = input("Enter language for translation: ")
            translation = input("Enter translation: ")
            
            # Check if phrase exists
            if english not in phrases:
                phrases[english] = {}
            phrases[english][language] = translation
            savePhrases(phrases)
            print("Phrase added successfully!")
        
        
        elif choice == '3':
            # Translate a phrase
            print("\nAvailable phrases:")
            for i, phrase in enumerate(phrases.keys(), 1):
                print(f"{i}. {phrase}")
            
           
            # Print the phrase translations
            try:
                phrase_num = int(input("\nEnter phrase number: ")) - 1
                phrase = list(phrases.keys())[phrase_num]
                
                print(f"\nAvailable translations for '{phrase}':")
                for language in phrases[phrase]:
                    print(f"{language}: {phrases[phrase][language]}")
           
            except:
                print("Invalid input!")
        
        
        elif choice == '4':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice! Please try again.")

main()