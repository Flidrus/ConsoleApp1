using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Embeddings;
using Microsoft.SemanticKernel.ChatCompletion;

namespace SearchApp
{
    class Program
    {
        static async Task Main(string[] args)
        {
            Console.OutputEncoding = Encoding.UTF8;
            Console.InputEncoding = Encoding.UTF8;

            string ollamaEndpoint = "http://localhost:11434";
            string qdrantEndpoint = "http://localhost:6333";
            string collectionName = "azure_all_labs_index";

            using var qdrantHttpClient = new HttpClient { Timeout = TimeSpan.FromMinutes(5) };
            using var ollamaHttpClient = new HttpClient { BaseAddress = new Uri(ollamaEndpoint), Timeout = TimeSpan.FromMinutes(10) };

            Console.WriteLine("=== SEARCH APP: INITIALIZING SEMANTIC KERNEL ===");
            var kernelBuilder = Kernel.CreateBuilder();

            // Нам потрібні і вектори (для пошуку), і чат (для відповіді)
            kernelBuilder.AddOllamaTextEmbeddingGeneration("nomic-embed-text", httpClient: ollamaHttpClient);
            kernelBuilder.AddOllamaChatCompletion("tinyllama", httpClient: ollamaHttpClient);

            var kernel = kernelBuilder.Build();
            var embeddingService = kernel.GetRequiredService<ITextEmbeddingGenerationService>();
            var chatService = kernel.GetRequiredService<IChatCompletionService>();

            Console.WriteLine("\n=== STEP 1: USER QUERY & QDRANT RETRIEVAL ===");
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("ENTER YOUR QUERY:");
            Console.ResetColor();
            Console.Write("\nUser: ");
            string userQuery = Console.ReadLine() ?? "Which method is responsible for translating text and calls TranslateTextAsync?";

            // Генеруємо вектор з питання
            var queryVector = await embeddingService.GenerateEmbeddingAsync(userQuery);

            var searchBody = new { vector = queryVector.ToArray(), limit = 2, with_payload = true };
            var searchContent = new StringContent(JsonSerializer.Serialize(searchBody), Encoding.UTF8, "application/json");

            // Шукаємо в існуючій базі даних
            var searchResponse = await qdrantHttpClient.PostAsync($"{qdrantEndpoint}/collections/{collectionName}/points/search", searchContent);
            var searchResultJson = await searchResponse.Content.ReadAsStringAsync();

            string retrievedContext = "";
            try
            {
                using var doc = JsonDocument.Parse(searchResultJson);
                var resultRoot = doc.RootElement.GetProperty("result");
                foreach (var item in resultRoot.EnumerateArray())
                {
                    retrievedContext += item.GetProperty("payload").GetProperty("text").GetString() + "\n---\n";
                }
            }
            catch { retrievedContext = "CONTEXT NOT FOUND."; }

            Console.ForegroundColor = ConsoleColor.DarkGray;
            Console.WriteLine("\n--- ЗНАЙДЕНИЙ КОНТЕКСТ ДЛЯ LLM ---");
            Console.WriteLine(retrievedContext);
            Console.WriteLine("----------------------------------");
            Console.ResetColor();

            Console.WriteLine("\n=== STEP 2: FORMULATING PROMPT & CALLING LLM ===");
            string systemInstruction = "You are an assistant. Answer the user's question using ONLY the provided CODE CONTEXT. If the answer is not present in the CODE CONTEXT, output exactly 'I don't know'. Be short and direct.";
            string combinedUserMessage = $"CODE CONTEXT:\n{retrievedContext}\nQUESTION: {userQuery}";

            var chatHistory = new ChatHistory();
            chatHistory.AddSystemMessage(systemInstruction);
            chatHistory.AddUserMessage(combinedUserMessage);

            Console.WriteLine("Sending prompt to local LLM (TinyLlama)... Please wait.");
            Console.Write("\nTinyLlama: ");

            using var cts = new System.Threading.CancellationTokenSource(TimeSpan.FromMinutes(10));
            var executionSettings = new PromptExecutionSettings { ExtensionData = new Dictionary<string, object> { { "temperature", 0.0 }, { "top_p", 0.1 } } };

            try
            {
                var chatResponse = await chatService.GetChatMessageContentAsync(chatHistory, executionSettings: executionSettings, cancellationToken: cts.Token);
                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine(chatResponse.Content);
            }
            catch (Exception ex)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"\n[LLM Error]: {ex.Message}");
            }

            Console.ResetColor();
            Console.WriteLine("\nPress any key to exit...");
            Console.ReadKey();
        }
    }
}