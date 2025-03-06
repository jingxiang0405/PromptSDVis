import openai
from openai import OpenAI
"""
def check_openai_api_key(api_key):
    openai.api_key = ""
    try:
        print("openai.Model.list()")
        #print(openai.Model.list())
    except openai.error.AuthenticationError as e:
        return False
    else:
        return True


api_key = ""
is_valid = check_openai_api_key(api_key)

if is_valid:
    print("Valid OpenAI API key.")
else:
    print("Invalid OpenAI API key.")
"""
"""
openai.api_key =""
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages = [
    {
        'role': 'user',
        'content': 'You are an expert in named entity recognition. You are good at information extraction. Now we will perform customized named entity recognition and first explain all entities.\nThere are five entities that need to be recognized: Subject term, Style modifier, Image prompt, Quality booster, Repeating term, and Magic term. The definitions of these five entities are as follows:\nSubject term: Denotes the subject\nStyle modifier: Indicates an artistic style\nImage prompt: Indicates a style or subject via an image\nQuality booster: A term intended to improve the quality of the image\nRepeating term: Repetition of subject terms or style terms with the intention of strengthening this subject or style\nMagic term: A term that is semantically different from the rest of the prompt with the intention to produce surprising results\nHere are the detailed explanations:\nDetailed Description\nSubject Terms:\nIndicate the desired subject to the text-to-image system (e.g., “a landscape” or “an old car in a meadow”).\nEssential for controlling the image generation process.\nExample issue: Early systems struggled with specific subjects without titles, like Zdzisław Beksiński’s artworks.\nStyle Modifiers:\nAdded to produce images in a certain style or artistic medium.\nExamples: “oil painting,” “by Francisco Goya,” “hyperrealistic,” “Cubism.”\nCan include art periods, schools, styles, materials, media, techniques, and artists.\nImage Prompts:\nProvide a visual target for the synthesis of the image in terms of style and subject.\nSpecified as URLs in the textual input prompt.\nDifferent from “initial images” which enhance or distort the starting image.\nQuality Boosters:\nIncrease aesthetic qualities and detail level in images.\nExamples: “trending on artstation,” “highly detailed,” “epic,” “8k.”\nVerbosity in prompts may improve detail but reduce subject control.\nRepeating Terms:\nStrengthen the associations formed by the generative system.\nExample: “space whale. a whale in space” produces better results.\nRepeating terms cause the system to activate related neural network regions.\nMagic Terms:\nIntroduce randomness leading to surprising results.\nExample: “control the soul” added to a prompt for “more magic, more wizard-ish imagery.”\nIntroduce unpredictability and increase variation in output.\nCan refer to non-visual qualities like touch, hearing, smell, and taste.\nSubsequently, we will proceed with the task of recognizing these entities.\nGiven entity label set: [\'Subject term\', \'Style modifier\', \'Image prompt\', \'Quality booster\', \'Repeating term\', \'Magic term\']\nPlease recognize the named entities in the given text. Based on the given entity label set, provide answer in the following JSON format: [{"Entity Name": "Entity Label"}]. If there is no entity in the text, return the following empty list: [].\nText: character environment design, arrogant elegant man travels through time via steampunk portals, pixiv fanbox, dramatic lighting, maximalist pastel color palette, splatter paint, pixar and disney exploded - view drawing, graphic novel by fiona staples and dustin nguyen, peter elson, alan bean, wangechi mutu, clean cel shaded vector art, trending on artstation\nAnswer: '
    }
]

)

print(response['choices'])
"""


client = OpenAI(
    base_url = 'http://140.119.162.202/api/chat',
    api_key='', # required, but unused
)

response = client.chat.completions.create(
    model="gemma2:27b",
    messages=[

    {"role": "user", 
     'content': 'You are an expert in named entity recognition. You are good at information extraction. Now we will perform customized named entity recognition and first explain all entities.\nThere are five entities that need to be recognized: Subject term, Style modifier, Image prompt, Quality booster, Repeating term, and Magic term. The definitions of these five entities are as follows:\nSubject term: Denotes the subject\nStyle modifier: Indicates an artistic style\nImage prompt: Indicates a style or subject via an image\nQuality booster: A term intended to improve the quality of the image\nRepeating term: Repetition of subject terms or style terms with the intention of strengthening this subject or style\nMagic term: A term that is semantically different from the rest of the prompt with the intention to produce surprising results\nHere are the detailed explanations:\nDetailed Description\nSubject Terms:\nIndicate the desired subject to the text-to-image system (e.g., “a landscape” or “an old car in a meadow”).\nEssential for controlling the image generation process.\nExample issue: Early systems struggled with specific subjects without titles, like Zdzisław Beksiński’s artworks.\nStyle Modifiers:\nAdded to produce images in a certain style or artistic medium.\nExamples: “oil painting,” “by Francisco Goya,” “hyperrealistic,” “Cubism.”\nCan include art periods, schools, styles, materials, media, techniques, and artists.\nImage Prompts:\nProvide a visual target for the synthesis of the image in terms of style and subject.\nSpecified as URLs in the textual input prompt.\nDifferent from “initial images” which enhance or distort the starting image.\nQuality Boosters:\nIncrease aesthetic qualities and detail level in images.\nExamples: “trending on artstation,” “highly detailed,” “epic,” “8k.”\nVerbosity in prompts may improve detail but reduce subject control.\nRepeating Terms:\nStrengthen the associations formed by the generative system.\nExample: “space whale. a whale in space” produces better results.\nRepeating terms cause the system to activate related neural network regions.\nMagic Terms:\nIntroduce randomness leading to surprising results.\nExample: “control the soul” added to a prompt for “more magic, more wizard-ish imagery.”\nIntroduce unpredictability and increase variation in output.\nCan refer to non-visual qualities like touch, hearing, smell, and taste.\nSubsequently, we will proceed with the task of recognizing these entities.\nGiven entity label set: [\'Subject term\', \'Style modifier\', \'Image prompt\', \'Quality booster\', \'Repeating term\', \'Magic term\']\nPlease recognize the named entities in the given text. Based on the given entity label set, provide answer in the following JSON format: [{"Entity Name": "Entity Label"}]. If there is no entity in the text, return the following empty list: [].\nText: character environment design, arrogant elegant man travels through time via steampunk portals, pixiv fanbox, dramatic lighting, maximalist pastel color palette, splatter paint, pixar and disney exploded - view drawing, graphic novel by fiona staples and dustin nguyen, peter elson, alan bean, wangechi mutu, clean cel shaded vector art, trending on artstation\nAnswer: '
    },

    ]
)


"""

[<OpenAIObject at 0x103ce7270> JSON: {
  "index": 0,
  "message": {
    "role": "assistant",
    "content": "```json\n[\n    {\"Entity Name\": \"arrogant elegant man travels through time via steampunk portals\", \"Entity Label\": \"Subject term\"},\n    {\"Entity Name\": \"dramatic lighting\", \"Entity Label\": \"Style modifier\"},\n    {\"Entity Name\": \"maximalist pastel color palette\", \"Entity Label\": \"Style modifier\"},\n    {\"Entity Name\": \"splatter paint\", \"Entity Label\": \"Style modifier\"},\n    {\"Entity Name\": \"pixar and disney exploded - view drawing\", \"Entity Label\": \"Image prompt\"},\n    {\"Entity Name\": \"graphic novel by fiona staples and dustin nguyen\", \"Entity Label\": \"Image prompt\"},\n    {\"Entity Name\": \"peter elson\", \"Entity Label\": \"Image prompt\"},\n    {\"Entity Name\": \"alan bean\", \"Entity Label\": \"Image prompt\"},\n    {\"Entity Name\": \"wangechi mutu\", \"Entity Label\": \"Image prompt\"},\n    {\"Entity Name\": \"clean cel shaded vector art\", \"Entity Label\": \"Style modifier\"},\n    {\"Entity Name\": \"trending on artstation\", \"Entity Label\": \"Quality booster\"}\n]\n```",
    "refusal": null
  },
  "logprobs": null,
  "finish_reason": "stop"
}]
"""