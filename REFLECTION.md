# Reflection - LibraryMind

## Design Decisions

I used one gateway (Amali) to reach both OpenAI and Anthropic instead of wiring in two separate SDKs. This kept the provider classes small, they just format a request and call the same function underneath, and the retry logic only had to be written once. The downside is both providers share one API key and one request shape, so I cannot tune each provider on its own. That felt fair, since the goal was showing fallback work, not squeezing the most out of each provider.

For the RAG engine, I made the relevance threshold something you set in the .env file instead of a fixed number in the code, since I kept adjusting it while building. I also added a check that clears the sources list if the AI itself says it could not find anything, even when the search step already found matches above the threshold. Without that check, a patron could get an answer that says "I don't know" sitting right next to a list of book titles, which does not make sense to read.

The chat feature does not reuse the cache from the question answering engine, on purpose. That cache stores answers by the exact question text, fine for one question but wrong for a conversation, since two patrons asking the same first question could end up seeing each other's chat history. I copied a small amount of logic instead of risking that.

The last big decision was how the app builds its AI service. At first every service (RAG, chat, classification, summary) built its own copy of the same fallback logic, four copies of something that always ended up the same. I changed it so the app builds one shared copy and every service uses that, which removed repeated code and made clear this piece is meant to be shared, not rebuilt.

## Challenges

The hardest bug to catch was a mix up between distance and similarity. ChromaDB gives back a distance, where a lower number means the two things are more alike, but in my head I had it backwards, like a higher number meant a better match. That let bad matches in and dropped good ones at the same time. Once I looked at the actual numbers I fixed it by flipping the score (1 minus the distance) and left a note on the class so I do not repeat that mistake.

Getting clean JSON back from classification and summary was another repeat problem. The AI would sometimes wrap the JSON in a code block even though the prompt told it not to. I fixed this with a small function that strips those code fences before parsing, and if the JSON still does not parse, the error message includes the raw text the AI sent back, so I can see exactly what went wrong instead of guessing.

## A Debugging Story

After the RAG pipeline was working, I kept getting the "I couldn't find any books" message for questions that clearly had good answers, like asking about desert planets, which should point straight to Dune. I assumed the threshold was too strict, so I raised it a little, from 0.4 to 0.5. It still failed.

That is when I stopped guessing and actually measured what was happening. I wrote a small script that embedded a few real questions and printed the distance for every match ChromaDB returned, and the result surprised me. Even the correct answer for the desert planet question sat at a distance of about 0.47, and the exact phrase from the assignment brief, "space travel adventure," had its best match at 0.70. A genuinely off topic question like "what is the meaning of life" only dropped to about 0.79.

So the real problem was never that the threshold was too tight in the way I first assumed. My whole idea of what a good distance number looks like was wrong for this embedding model. I set the threshold to 0.73 based on the real numbers, checked it against good and bad questions, and it worked. The lesson was to stop tuning a number by feel and print out what the system is actually doing before changing anything.

## Extensions Attempted

Past what the assignment asked for, I added: a relevance threshold set through the environment instead of hardcoded, the source clearing check above, proper logging that turns on no matter how the server starts (before this fix it only worked by accident, from a default uvicorn sets, not anything I had set up), and a small set of custom error types so the API can tell "all AI providers failed" apart from some other unrelated crash, instead of catching every RuntimeError the same way. I also wrote more tests than required, over 180 in total, including a full set that checks the exact boundary of every field on every endpoint.

Testing against a real, live server instead of only mocked tests caught two bugs I would have missed otherwise. The review summary endpoint let you send any number of reviews with no upper limit, even though the brief caps it at 50, so I added that check. And the smoke test script crashed instantly on a plain Windows terminal, because it tried to print a character the default Windows console encoding cannot handle. Both were quick fixes, but I only found them by actually running the thing instead of trusting the code looked right.

I also tried to test provider fallback for real, by breaking the primary key and checking the backup answers instead. That does not work here, since both providers share one key through the gateway and it would break both at once. While digging into this I found Anthropic was genuinely failing on its own that day, a real gateway error, not something I caused. That let me prove fallback honestly: put the already broken provider first and confirm the working one still saves the request. It did.
