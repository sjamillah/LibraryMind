# Reflection - LibraryMind

## Design Decisions

I used one gateway (Amali) to reach both OpenAI and Anthropic instead of wiring in two separate SDKs. This meant the OpenAI and Anthropic provider classes could stay small and simple. They just format a request and call the same function underneath. The retry logic only had to be written once too. The downside is both providers share one API key and one request shape, so I cannot tune each provider on its own. For this project that felt like a fair trade. The goal was to show the fallback working, not to squeeze the most out of each provider.

For the RAG engine, I made the relevance threshold something you can set in the .env file instead of a fixed number buried in the code. I kept adjusting it while building, so it made sense to pull it out. I also added a check that clears the sources list if the AI itself says it could not find anything, even when the search step already found matches above the threshold. Without that check, a patron could get an answer that says "I don't know" sitting right next to a list of book titles, which does not make sense to read.

The chat feature does not reuse the cache from the question answering engine, on purpose. That cache stores answers by the exact question text, which is fine for a single question but wrong for a conversation. If it reused that cache, two different patrons asking the same first question could end up seeing each other's chat history. I copied a small amount of logic instead of risking that.

The last big decision was about how the app builds its AI service. At first, every service (RAG, chat, classification, summary) built its own copy of the same fallback logic. That was four copies of something that always ended up the same anyway. I changed it so the app builds one shared copy and every service uses that same one. This did not change how anything behaves, it just removed repeated code and made it clear that this piece is meant to be shared, not rebuilt.

## Challenges

The hardest bug to catch was a mix up between distance and similarity. ChromaDB gives back a distance, where a lower number means the two things are more alike. In my head I had it backwards, like a higher number meant a better match. That mistake changed how my threshold filtered results, so I was letting in some bad matches and dropping some good ones at the same time. Once I looked at the actual numbers I fixed it by flipping the score (1 minus the distance) and left a note next to the class so I do not make that same mistake again later.

Getting clean JSON back from the classification and summary features was another repeat problem. The AI would sometimes wrap the JSON in a code block even though the prompt told it not to. I fixed this with a small function that strips those code fences before trying to read the JSON. If the JSON still does not parse, the error message includes the raw text the AI sent back, so if it breaks again I can see exactly what went wrong instead of guessing.

## A Debugging Story

After I had the whole RAG pipeline working, I kept getting the "I couldn't find any books" message for questions that clearly had good answers in the catalogue, like asking about desert planets, which should point straight to Dune. I assumed the relevance threshold was too strict, so I raised it a little, from 0.4 to 0.5. It still failed.

That is when I stopped guessing and actually measured what was happening. I wrote a small script that embedded a few real questions and printed the distance for every match ChromaDB returned. The result surprised me. Even the correct answer for the desert planet question had a distance of about 0.47, and the exact phrase used in the assignment brief, "space travel adventure," had its best match sitting at 0.70. Meanwhile a genuinely off topic question like "what is the meaning of life" only dropped to about 0.79 at its closest match.

So the real problem was never that the threshold was too tight in the way I first assumed. My whole idea of what a good distance number looks like was wrong for this embedding model. I set the threshold to 0.73 based on the real numbers instead of a guess, checked it against both good and bad questions, and it worked. The lesson for me was to stop tuning a number by feel and instead print out what the system is actually doing before changing anything.

## Extensions Attempted

Past what the assignment asked for, I added a few things: a relevance threshold you can change through the environment instead of hardcoding it, the source clearing check described above, proper logging that turns on no matter how the server is started (before this fix, logging only worked by accident, because of a default that uvicorn sets, not because I had set it up myself), and a small set of custom error types so the API can tell the difference between "all AI providers failed" and some other unrelated crash, instead of catching every RuntimeError the same way. I also wrote more tests than the assignment required, well over 135 in total, covering things like provider fallback, cache misses, rate limits, and the edge cases around the RAG threshold.
