"use strict";

// Paper and model links are added when those releases are available.
window.AGF_LINKS = { arxiv: "", huggingface: "", code: "https://github.com/wuli-art/Adversarial-Gradient-Forcing", project: "https://wuli-art.github.io/Adversarial-Gradient-Forcing/" };

window.AGF_CASES = [
  {
    "id": "049",
    "title": "A little yawn",
    "category": "People",
    "prompt": "A candid moment captured in a soft, natural light photograph of a little child yawning widely. The child has rosy cheeks and tousled brown hair, with large, curious eyes that gaze directly into the camera. They sit on a cozy wooden floor, surrounded by colorful toys and books. The background features a warm, inviting room with a soft rug and a few stuffed animals nearby. The yawn stretches the child's small face, revealing a hint of fatigue. The photo has a gentle, documentary style. A close-up shot with a slight tilt, capturing the child's entire body and the surrounding environment.",
    "videos": {
      "agf": "assets/videos/049/agf.mp4",
      "sgf": "assets/videos/049/sgf.mp4",
      "opsdv": "assets/videos/049/opsdv.mp4",
      "self-forcing": "assets/videos/049/self-forcing.mp4",
      "causal-forcing": "assets/videos/049/causal-forcing.mp4",
      "longlive": "assets/videos/049/longlive.mp4"
    },
    "posters": {
      "agf": "assets/posters/049-agf.jpg",
      "sgf": "assets/posters/049-sgf.jpg",
      "opsdv": "assets/posters/049-opsdv.jpg",
      "self-forcing": "assets/posters/049-self-forcing.jpg",
      "causal-forcing": "assets/posters/049-causal-forcing.jpg",
      "longlive": "assets/posters/049-longlive.jpg"
    }
  },
  {
    "id": "087",
    "title": "The salad toss",
    "category": "People",
    "prompt": "A dynamic scene captured in the style of a vibrant food photography shoot, showcasing a chef expertly tossing a salad in a large ceramic bowl. The chef, with a lively expression and focused intensity, moves with grace and precision, the salad spinning gracefully in the air before landing back in the bowl with a satisfying clatter. The chef is dressed in a crisp white chef's coat and black pants, with a white hat perched on his head. The background is a clean, modern kitchen with stainless steel appliances and a backdrop of warm, soft lighting that highlights the freshness of the ingredients. A mid-shot from a slightly elevated angle, capturing both the chef's action and the vibrant salad.",
    "videos": {
      "agf": "assets/videos/087/agf.mp4",
      "sgf": "assets/videos/087/sgf.mp4",
      "opsdv": "assets/videos/087/opsdv.mp4",
      "self-forcing": "assets/videos/087/self-forcing.mp4",
      "causal-forcing": "assets/videos/087/causal-forcing.mp4",
      "longlive": "assets/videos/087/longlive.mp4"
    },
    "posters": {
      "agf": "assets/posters/087-agf.jpg",
      "sgf": "assets/posters/087-sgf.jpg",
      "opsdv": "assets/posters/087-opsdv.jpg",
      "self-forcing": "assets/posters/087-self-forcing.jpg",
      "causal-forcing": "assets/posters/087-causal-forcing.jpg",
      "longlive": "assets/posters/087-longlive.jpg"
    }
  },
  {
    "id": "012",
    "title": "Pirate ships in a coffee cup",
    "category": "Imagination",
    "prompt": "A photorealistic closeup video of two pirate ships battling each other as they sail inside a steaming cup of coffee. The ships are intricately detailed, with wooden planks, sails flapping in the breeze, and cannons aimed at each other. The crew members, wearing authentic pirate attire, brandish swords and pistols, their expressions fierce and determined. The coffee foam creates a frothy, turbulent sea, with ripples and waves realistically depicted. The background is a blurred, warm brown coffee surface, with steam rising gently. The camera angle is slightly elevated, capturing the intense action from above.",
    "videos": {
      "agf": "assets/videos/012/agf.mp4",
      "sgf": "assets/videos/012/sgf.mp4",
      "opsdv": "assets/videos/012/opsdv.mp4",
      "self-forcing": "assets/videos/012/self-forcing.mp4",
      "causal-forcing": "assets/videos/012/causal-forcing.mp4",
      "longlive": "assets/videos/012/longlive.mp4"
    },
    "posters": {
      "agf": "assets/posters/012-agf.jpg",
      "sgf": "assets/posters/012-sgf.jpg",
      "opsdv": "assets/posters/012-opsdv.jpg",
      "self-forcing": "assets/posters/012-self-forcing.jpg",
      "causal-forcing": "assets/posters/012-causal-forcing.jpg",
      "longlive": "assets/posters/012-longlive.jpg"
    }
  },
  {
    "id": "032",
    "title": "A robot in the neon city",
    "category": "Imagination",
    "prompt": "A cyberpunk-style illustration depicting a lone robot navigating a neon-lit cityscape. The robot stands tall with sleek, metallic armor, adorned with blinking lights and wires. Its eyes, glowing with a deep blue hue, scan the surroundings with curiosity. The background features towering skyscrapers, holographic advertisements, and crowded streets filled with various cyborgs and humans. The air is thick with smoke and the hum of technology. A medium shot from a high-angle perspective, capturing both the robot and the bustling city environment.",
    "videos": {
      "agf": "assets/videos/032/agf.mp4",
      "sgf": "assets/videos/032/sgf.mp4",
      "opsdv": "assets/videos/032/opsdv.mp4",
      "self-forcing": "assets/videos/032/self-forcing.mp4",
      "causal-forcing": "assets/videos/032/causal-forcing.mp4",
      "longlive": "assets/videos/032/longlive.mp4"
    },
    "posters": {
      "agf": "assets/posters/032-agf.jpg",
      "sgf": "assets/posters/032-sgf.jpg",
      "opsdv": "assets/posters/032-opsdv.jpg",
      "self-forcing": "assets/posters/032-self-forcing.jpg",
      "causal-forcing": "assets/posters/032-causal-forcing.jpg",
      "longlive": "assets/posters/032-longlive.jpg"
    }
  },
  {
    "id": "027",
    "title": "Flowers reclaim a warehouse",
    "category": "Nature",
    "prompt": "A dramatic and surreal photograph in a realistic style, capturing an abandoned warehouse where vibrant flowers are blooming from the cracked concrete walls. The flowers are diverse, ranging from wild daisies to delicate roses, their colors vivid and varied. The space is dimly lit, with shadows cast by the uneven concrete surfaces. The camera angle is low, emphasizing the growth and生命力, with a sense of nature reclaiming the urban environment. The background shows the remnants of old machinery and graffiti, adding to the desolate yet hopeful atmosphere. A close-up shot from a slightly downward angle, highlighting the contrast between the harsh industrial setting and the blooming flowers.",
    "videos": {
      "agf": "assets/videos/027/agf.mp4",
      "sgf": "assets/videos/027/sgf.mp4",
      "opsdv": "assets/videos/027/opsdv.mp4",
      "self-forcing": "assets/videos/027/self-forcing.mp4",
      "causal-forcing": "assets/videos/027/causal-forcing.mp4",
      "longlive": "assets/videos/027/longlive.mp4"
    },
    "posters": {
      "agf": "assets/posters/027-agf.jpg",
      "sgf": "assets/posters/027-sgf.jpg",
      "opsdv": "assets/posters/027-opsdv.jpg",
      "self-forcing": "assets/posters/027-self-forcing.jpg",
      "causal-forcing": "assets/posters/027-causal-forcing.jpg",
      "longlive": "assets/posters/027-longlive.jpg"
    }
  },
  {
    "id": "107",
    "title": "Opening a holiday gift",
    "category": "People",
    "prompt": "A vibrant and dynamic digital illustration in the style of a modern holiday card, featuring a young woman with sparkling eyes and a radiant smile as she opens a gift. She has wavy brown hair and fair skin, standing in a cozy living room with soft lighting. The gift is wrapped in bright red paper with gold ribbons, and the woman's fingers gently pull back the paper, revealing the contents inside. The background includes a Christmas tree with twinkling lights and some ornaments, along with a fireplace with warm embers glowing. A close-up shot from a slightly above angle, capturing the woman's joy and anticipation.",
    "videos": {
      "agf": "assets/videos/107/agf.mp4",
      "sgf": "assets/videos/107/sgf.mp4",
      "opsdv": "assets/videos/107/opsdv.mp4",
      "self-forcing": "assets/videos/107/self-forcing.mp4",
      "causal-forcing": "assets/videos/107/causal-forcing.mp4",
      "longlive": "assets/videos/107/longlive.mp4"
    },
    "posters": {
      "agf": "assets/posters/107-agf.jpg",
      "sgf": "assets/posters/107-sgf.jpg",
      "opsdv": "assets/posters/107-opsdv.jpg",
      "self-forcing": "assets/posters/107-self-forcing.jpg",
      "causal-forcing": "assets/posters/107-causal-forcing.jpg",
      "longlive": "assets/posters/107-longlive.jpg"
    }
  },
  {
    "id": "118",
    "title": "A dance in watercolor",
    "category": "Animation",
    "prompt": "A whimsical illustration in a vibrant watercolor style depicting two blobs in a passionate dance of love. One blob is green, with leaves and vines swirling around it, giving it a lively and organic feel. The other blob is orange, adorned with sunbursts and warm hues, symbolizing warmth and energy. They twirl gracefully, their forms blending harmoniously. The background is a soft, pastel gradient with gentle swirls and patterns, enhancing the dreamy and romantic atmosphere. A dynamic, mid-shot from a slightly elevated angle captures the intimate moment.",
    "videos": {
      "agf": "assets/videos/118/agf.mp4",
      "sgf": "assets/videos/118/sgf.mp4",
      "opsdv": "assets/videos/118/opsdv.mp4",
      "self-forcing": "assets/videos/118/self-forcing.mp4",
      "causal-forcing": "assets/videos/118/causal-forcing.mp4",
      "longlive": "assets/videos/118/longlive.mp4"
    },
    "posters": {
      "agf": "assets/posters/118-agf.jpg",
      "sgf": "assets/posters/118-sgf.jpg",
      "opsdv": "assets/posters/118-opsdv.jpg",
      "self-forcing": "assets/posters/118-self-forcing.jpg",
      "causal-forcing": "assets/posters/118-causal-forcing.jpg",
      "longlive": "assets/posters/118-longlive.jpg"
    }
  },
  {
    "id": "035",
    "title": "Dancing felt creatures",
    "category": "Animation",
    "prompt": "A miniature 3D render in an octane engine style depicting adorable wool and felt monsters dancing together in a dreamy, bokeh-filled setting. These soft, cuddly creatures, with big expressive eyes and fluffy bodies, are illuminated by gentle, diffused lighting that casts a warm, ethereal glow. The background features a soft, hazy backdrop with a dreamy bokeh effect, adding a cinematic quality to the scene. The monsters are shown from various angles, capturing their playful movements and expressions, creating a charming and enchanting atmosphere. A medium shot with a dynamic camera angle, highlighting the natural and joyful dance of these woolen monsters.",
    "videos": {
      "agf": "assets/videos/035/agf.mp4",
      "sgf": "assets/videos/035/sgf.mp4",
      "opsdv": "assets/videos/035/opsdv.mp4",
      "self-forcing": "assets/videos/035/self-forcing.mp4",
      "causal-forcing": "assets/videos/035/causal-forcing.mp4",
      "longlive": "assets/videos/035/longlive.mp4"
    },
    "posters": {
      "agf": "assets/posters/035-agf.jpg",
      "sgf": "assets/posters/035-sgf.jpg",
      "opsdv": "assets/posters/035-opsdv.jpg",
      "self-forcing": "assets/posters/035-self-forcing.jpg",
      "causal-forcing": "assets/posters/035-causal-forcing.jpg",
      "longlive": "assets/posters/035-longlive.jpg"
    }
  },
  {
    "id": "017",
    "title": "A creature by candlelight",
    "category": "Animation",
    "prompt": "A close-up 3D animated scene of a short, fluffy monster kneeling beside a melting red candle. The monster has large, wide eyes and an open mouth, gazing at the flame with a look of wonder and curiosity. Its soft, fluffy fur contrasts with the warm, dramatic lighting that highlights every detail of its gentle, innocent expression. The pose conveys a sense of playfulness and exploration, as if the creature is discovering the world for the first time. The background features a cozy, warmly lit room with subtle hints of a fireplace and soft furnishings, enhancing the overall atmosphere. The use of warm colors and dramatic lighting creates a captivating and inviting scene.",
    "videos": {
      "agf": "assets/videos/017/agf.mp4",
      "sgf": "assets/videos/017/sgf.mp4",
      "opsdv": "assets/videos/017/opsdv.mp4",
      "self-forcing": "assets/videos/017/self-forcing.mp4",
      "causal-forcing": "assets/videos/017/causal-forcing.mp4",
      "longlive": "assets/videos/017/longlive.mp4"
    },
    "posters": {
      "agf": "assets/posters/017-agf.jpg",
      "sgf": "assets/posters/017-sgf.jpg",
      "opsdv": "assets/posters/017-opsdv.jpg",
      "self-forcing": "assets/posters/017-self-forcing.jpg",
      "causal-forcing": "assets/posters/017-causal-forcing.jpg",
      "longlive": "assets/posters/017-longlive.jpg"
    }
  },
  {
    "id": "112",
    "title": "A panda in the laboratory",
    "category": "Animation",
    "prompt": "A scientific laboratory scene in a detailed digital painting style, featuring a panda wearing a white lab coat with the word \"LABORATORY\" emblazoned on the chest. The panda is intently working, holding a beaker in one hand and a test tube in the other, both filled with colorful liquids. It has a curious and focused expression, with large black circles surrounding bright brown eyes. The background showcases a cluttered laboratory with various scientific equipment, petri dishes, and books scattered about. Shelves lined with vials and chemicals add to the academic atmosphere. The lighting is soft yet precise, highlighting the panda's fur and the intricate details of the lab. A close-up medium shot from a slightly elevated angle, capturing the panda's detailed face and the bustling laboratory environment.",
    "videos": {
      "agf": "assets/videos/112/agf.mp4",
      "sgf": "assets/videos/112/sgf.mp4",
      "opsdv": "assets/videos/112/opsdv.mp4",
      "self-forcing": "assets/videos/112/self-forcing.mp4",
      "causal-forcing": "assets/videos/112/causal-forcing.mp4",
      "longlive": "assets/videos/112/longlive.mp4"
    },
    "posters": {
      "agf": "assets/posters/112-agf.jpg",
      "sgf": "assets/posters/112-sgf.jpg",
      "opsdv": "assets/posters/112-opsdv.jpg",
      "self-forcing": "assets/posters/112-self-forcing.jpg",
      "causal-forcing": "assets/posters/112-causal-forcing.jpg",
      "longlive": "assets/posters/112-longlive.jpg"
    }
  },
  {
    "id": "096",
    "title": "Mushrooms among forgotten books",
    "category": "Nature",
    "prompt": "A surreal and haunting digital art piece in a dreamlike style, featuring mushrooms sprouting from the base of a decaying bookshelf. The mushrooms have vibrant, colorful caps in shades of orange, yellow, and green, contrasting sharply with the worn, weathered wood of the bookshelf. The bookshelf is covered in dust and peeling paint, with several books lying open and pages torn. The background is dimly lit, with flickering light casting shadows and highlighting the decay. The mushrooms appear to be growing from cracks and crevices in the wood, giving the scene a mysterious and eerie feel. A close-up shot from a slightly elevated angle, emphasizing the textures and colors.",
    "videos": {
      "agf": "assets/videos/096/agf.mp4",
      "sgf": "assets/videos/096/sgf.mp4",
      "opsdv": "assets/videos/096/opsdv.mp4",
      "self-forcing": "assets/videos/096/self-forcing.mp4",
      "causal-forcing": "assets/videos/096/causal-forcing.mp4",
      "longlive": "assets/videos/096/longlive.mp4"
    },
    "posters": {
      "agf": "assets/posters/096-agf.jpg",
      "sgf": "assets/posters/096-sgf.jpg",
      "opsdv": "assets/posters/096-opsdv.jpg",
      "self-forcing": "assets/posters/096-self-forcing.jpg",
      "causal-forcing": "assets/posters/096-causal-forcing.jpg",
      "longlive": "assets/posters/096-longlive.jpg"
    }
  },
  {
    "id": "080",
    "title": "Seahorses in the seagrass",
    "category": "Nature",
    "prompt": "A vibrant underwater scene in the style of a marine biology illustration, featuring a trio of seahorses gracefully holding onto seagrass with their tails. Each seahorse has a distinctive pattern on its body, ranging from deep blues and greens to lighter aquas and whites. Their tails wrap tightly around the swaying seagrass, which moves gently in the current. The seahorses have expressive eyes and small, delicate fins that flutter softly. The background showcases a rich variety of marine life, including colorful coral and various fish swimming around. The water is clear and filled with tiny bubbles rising to the surface. A close-up shot from a slightly elevated angle, capturing the intricate details of the seahorses and their environment.",
    "videos": {
      "agf": "assets/videos/080/agf.mp4",
      "sgf": "assets/videos/080/sgf.mp4",
      "opsdv": "assets/videos/080/opsdv.mp4",
      "self-forcing": "assets/videos/080/self-forcing.mp4",
      "causal-forcing": "assets/videos/080/causal-forcing.mp4",
      "longlive": "assets/videos/080/longlive.mp4"
    },
    "posters": {
      "agf": "assets/posters/080-agf.jpg",
      "sgf": "assets/posters/080-sgf.jpg",
      "opsdv": "assets/posters/080-opsdv.jpg",
      "self-forcing": "assets/posters/080-self-forcing.jpg",
      "causal-forcing": "assets/posters/080-causal-forcing.jpg",
      "longlive": "assets/posters/080-longlive.jpg"
    }
  },
  {
    "id": "111",
    "title": "Sculpting in clay",
    "category": "People",
    "prompt": "A clay model being slowly deformed as it is pressed and molded into a new shape by hand. The clay is a rich brown color, and the model, originally a simple figure, is gradually taking on a more complex form. The sculptor, a middle-aged man with weathered hands and focused expression, gently presses and molds the clay with precision. His movements are deliberate and steady, and the clay yields to his touch, revealing intricate details like folds and textures. The background is a dimly lit studio with shelves filled with various tools and other clay models. The camera angle is from the side, capturing both the sculptor's hands and the transformation of the clay. A close-up shot with a slight tilt to emphasize the process.",
    "videos": {
      "agf": "assets/videos/111/agf.mp4",
      "sgf": "assets/videos/111/sgf.mp4",
      "opsdv": "assets/videos/111/opsdv.mp4",
      "self-forcing": "assets/videos/111/self-forcing.mp4",
      "causal-forcing": "assets/videos/111/causal-forcing.mp4",
      "longlive": "assets/videos/111/longlive.mp4"
    },
    "posters": {
      "agf": "assets/posters/111-agf.jpg",
      "sgf": "assets/posters/111-sgf.jpg",
      "opsdv": "assets/posters/111-opsdv.jpg",
      "self-forcing": "assets/posters/111-self-forcing.jpg",
      "causal-forcing": "assets/posters/111-causal-forcing.jpg",
      "longlive": "assets/posters/111-longlive.jpg"
    }
  },
  {
    "id": "103",
    "title": "One delicious bite",
    "category": "People",
    "prompt": "A Chinese boy wearing glasses sits in a fast food restaurant, enjoying a delicious cheeseburger with his eyes closed. His hair is neatly combed, and he has a slightly dreamy expression. He holds the cheeseburger with both hands, taking a big bite. The background shows other diners and a colorful menu board with various fast food items. The lighting is warm and inviting, creating a cozy atmosphere. A close-up shot from a slightly lower angle, capturing the boy's joyful moment.",
    "videos": {
      "agf": "assets/videos/103/agf.mp4",
      "sgf": "assets/videos/103/sgf.mp4",
      "opsdv": "assets/videos/103/opsdv.mp4",
      "self-forcing": "assets/videos/103/self-forcing.mp4",
      "causal-forcing": "assets/videos/103/causal-forcing.mp4",
      "longlive": "assets/videos/103/longlive.mp4"
    },
    "posters": {
      "agf": "assets/posters/103-agf.jpg",
      "sgf": "assets/posters/103-sgf.jpg",
      "opsdv": "assets/posters/103-opsdv.jpg",
      "self-forcing": "assets/posters/103-self-forcing.jpg",
      "causal-forcing": "assets/posters/103-causal-forcing.jpg",
      "longlive": "assets/posters/103-longlive.jpg"
    }
  },
  {
    "id": "030",
    "title": "A truck above the city",
    "category": "Imagination",
    "prompt": "An extreme wide low-angle establishing shot from street level at dusk, capturing a surreal and unsettling scene. High above the ground, a garbage truck is floating and spinning, defying gravity. Garbage spills out of it, creating a chaotic whirlwind of debris. The cityscape below is dimly lit, with buildings and streetlights casting long shadows. The sky is a mix of deep purples and oranges, adding to the eerie atmosphere. The photo has a grainy, almost dreamlike quality.",
    "videos": {
      "agf": "assets/videos/030/agf.mp4",
      "sgf": "assets/videos/030/sgf.mp4",
      "opsdv": "assets/videos/030/opsdv.mp4",
      "self-forcing": "assets/videos/030/self-forcing.mp4",
      "causal-forcing": "assets/videos/030/causal-forcing.mp4",
      "longlive": "assets/videos/030/longlive.mp4"
    },
    "posters": {
      "agf": "assets/posters/030-agf.jpg",
      "sgf": "assets/posters/030-sgf.jpg",
      "opsdv": "assets/posters/030-opsdv.jpg",
      "self-forcing": "assets/posters/030-self-forcing.jpg",
      "causal-forcing": "assets/posters/030-causal-forcing.jpg",
      "longlive": "assets/posters/030-longlive.jpg"
    }
  },
  {
    "id": "033",
    "title": "A ninja in the desert",
    "category": "Imagination",
    "prompt": "A dramatic manga-style illustration of a stealthy ninja wandering through a vast, sun-baked desert. The ninja, dressed in black with a hood concealing most of their face, carries a large wooden case of wine slung over one shoulder. His movements are fluid and purposeful, with a slight lean forward as he walks. Behind him, a pack of hungry hyenas with sharp fangs and piercing eyes follow closely, snarling and drooling. The desert landscape is rugged, with dunes stretching to the horizon under a clear, hot sky. The background is filled with sparse cacti and rocks, creating a stark and ominous environment. The ninja's eyes are hidden behind shadows, hinting at a mysterious and dangerous journey. A dynamic mid-shot with the ninja leading the composition, the hyenas off to the side, and the desert stretching into the distance.",
    "videos": {
      "agf": "assets/videos/033/agf.mp4",
      "sgf": "assets/videos/033/sgf.mp4",
      "opsdv": "assets/videos/033/opsdv.mp4",
      "self-forcing": "assets/videos/033/self-forcing.mp4",
      "causal-forcing": "assets/videos/033/causal-forcing.mp4",
      "longlive": "assets/videos/033/longlive.mp4"
    },
    "posters": {
      "agf": "assets/posters/033-agf.jpg",
      "sgf": "assets/posters/033-sgf.jpg",
      "opsdv": "assets/posters/033-opsdv.jpg",
      "self-forcing": "assets/posters/033-self-forcing.jpg",
      "causal-forcing": "assets/posters/033-causal-forcing.jpg",
      "longlive": "assets/posters/033-longlive.jpg"
    }
  },
  {
    "id": "034",
    "title": "Hearing the news",
    "category": "People",
    "prompt": "A realistic photograph capturing a middle-aged man with a furrowed brow, his forehead creased with worry as he listens intently to some bad news. His eyes are wide and slightly teary, reflecting a mix of concern and distress. He has a weathered face with a few wrinkles around the mouth and eyes, suggesting years of hard work and worry. His suit is slightly rumpled, indicating the stress of the moment. The background is a cluttered office with papers scattered across a desk and a framed family photo leaning against a wall. The lighting is dim, casting shadows that add to the somber mood. A medium shot with the man looking directly at the camera, taken from a slight angle.",
    "videos": {
      "agf": "assets/videos/034/agf.mp4",
      "sgf": "assets/videos/034/sgf.mp4",
      "opsdv": "assets/videos/034/opsdv.mp4",
      "self-forcing": "assets/videos/034/self-forcing.mp4",
      "causal-forcing": "assets/videos/034/causal-forcing.mp4",
      "longlive": "assets/videos/034/longlive.mp4"
    },
    "posters": {
      "agf": "assets/posters/034-agf.jpg",
      "sgf": "assets/posters/034-sgf.jpg",
      "opsdv": "assets/posters/034-opsdv.jpg",
      "self-forcing": "assets/posters/034-self-forcing.jpg",
      "causal-forcing": "assets/posters/034-causal-forcing.jpg",
      "longlive": "assets/posters/034-longlive.jpg"
    }
  },
  {
    "id": "000",
    "title": "A walk through neon Tokyo",
    "category": "People",
    "prompt": "A stylish woman strolls down a bustling Tokyo street, the warm glow of neon lights and animated city signs casting vibrant reflections. She wears a sleek black leather jacket paired with a flowing red dress and black boots, her black purse slung over her shoulder. Sunglasses perched on her nose and a bold red lipstick add to her confident, casual demeanor. The street is damp and reflective, creating a mirror-like effect that enhances the colorful lights and shadows. Pedestrians move about, adding to the lively atmosphere. The scene is captured in a dynamic medium shot with the woman walking slightly to one side, highlighting her graceful strides.",
    "videos": {
      "agf": "assets/videos/000/agf.mp4",
      "sgf": "assets/videos/000/sgf.mp4",
      "opsdv": "assets/videos/000/opsdv.mp4",
      "self-forcing": "assets/videos/000/self-forcing.mp4",
      "causal-forcing": "assets/videos/000/causal-forcing.mp4",
      "longlive": "assets/videos/000/longlive.mp4"
    },
    "posters": {
      "agf": "assets/posters/000-agf.jpg",
      "sgf": "assets/posters/000-sgf.jpg",
      "opsdv": "assets/posters/000-opsdv.jpg",
      "self-forcing": "assets/posters/000-self-forcing.jpg",
      "causal-forcing": "assets/posters/000-causal-forcing.jpg",
      "longlive": "assets/posters/000-longlive.jpg"
    }
  },
  {
    "id": "022",
    "title": "A sunflower on the windowsill",
    "category": "Nature",
    "prompt": "A stop motion animation in a charming hand-drawn style, depicting a flower slowly growing out of the windowsill of a suburban house. The flower is a vibrant sunflower, with its petals unfurling gracefully. The windowsill is adorned with small potted plants and a few scattered books. The house has a cozy exterior, with a red door and white shutters, and the surrounding area features neatly trimmed bushes and a small garden path. The animation captures the natural growth process, with the sunflower stem bending slightly as it stretches upward. A close-up shot from a low angle, emphasizing the delicate details of the flower's growth.",
    "videos": {
      "agf": "assets/videos/022/agf.mp4",
      "sgf": "assets/videos/022/sgf.mp4",
      "opsdv": "assets/videos/022/opsdv.mp4",
      "self-forcing": "assets/videos/022/self-forcing.mp4",
      "causal-forcing": "assets/videos/022/causal-forcing.mp4",
      "longlive": "assets/videos/022/longlive.mp4"
    },
    "posters": {
      "agf": "assets/posters/022-agf.jpg",
      "sgf": "assets/posters/022-sgf.jpg",
      "opsdv": "assets/posters/022-opsdv.jpg",
      "self-forcing": "assets/posters/022-self-forcing.jpg",
      "causal-forcing": "assets/posters/022-causal-forcing.jpg",
      "longlive": "assets/posters/022-longlive.jpg"
    }
  },
  {
    "id": "048",
    "title": "The painter’s moving brush",
    "category": "People",
    "prompt": "A dynamic arc shot capturing a painter in front of a large canvas, swirling around to showcase their brush strokes from multiple angles. The painter, focused and engrossed, moves gracefully, their brush sweeping across the canvas with fluid motions. The canvas is filled with vibrant colors and intricate details, reflecting the artist's passionate and deliberate strokes. The background shows scattered paint tubes, brushes, and a palette, adding to the creative atmosphere. The lighting highlights the artist's movements and the textures on the canvas, creating a vivid and lively scene. A medium shot with a smooth circular motion, emphasizing the painter's technique and expression.",
    "videos": {
      "agf": "assets/videos/048/agf.mp4",
      "sgf": "assets/videos/048/sgf.mp4",
      "opsdv": "assets/videos/048/opsdv.mp4",
      "self-forcing": "assets/videos/048/self-forcing.mp4",
      "causal-forcing": "assets/videos/048/causal-forcing.mp4",
      "longlive": "assets/videos/048/longlive.mp4"
    },
    "posters": {
      "agf": "assets/posters/048-agf.jpg",
      "sgf": "assets/posters/048-sgf.jpg",
      "opsdv": "assets/posters/048-opsdv.jpg",
      "self-forcing": "assets/posters/048-self-forcing.jpg",
      "causal-forcing": "assets/posters/048-causal-forcing.jpg",
      "longlive": "assets/posters/048-longlive.jpg"
    }
  }
];
