def count(string):
   vowels=set("aeiouAEIOU")
   vowel_count=0
   consant_count=0
   for ch in string:
      if ch in vowels:
          vowel_count=vowel_count+1
      else:
          consant_count=consant_count+1
   return vowel_count,consant_count
string=input("Enter any string")
vowel_count,consant_count=count(string)
print("The number of vowels",vowel_count)
print("The number of consonants",consant_count)