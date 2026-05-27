using System;
using System.IO;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Embeddings;

namespace IndexerApp
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

            Console.WriteLine("=== INDEXER: INITIALIZING SEMANTIC KERNEL ===");
            var kernelBuilder = Kernel.CreateBuilder();
            // Нам потрібна ТІЛЬКИ генерація векторів (Embeddings), чат тут не потрібен
            kernelBuilder.AddOllamaTextEmbeddingGeneration("nomic-embed-text", httpClient: ollamaHttpClient);

            var kernel = kernelBuilder.Build();
            var embeddingService = kernel.GetRequiredService<ITextEmbeddingGenerationService>();

            Console.WriteLine("=== STEP 1: PREPARING QDRANT DATABASE ===");
            try
            {
                await qdrantHttpClient.DeleteAsync($"{qdrantEndpoint}/collections/{collectionName}");
                await Task.Delay(1000);

                var createReq = new { vectors = new { size = 768, distance = "Cosine" } };
                var content = new StringContent(JsonSerializer.Serialize(createReq), Encoding.UTF8, "application/json");
                var createRes = await qdrantHttpClient.PutAsync($"{qdrantEndpoint}/collections/{collectionName}", content);
                Console.WriteLine($"[Qdrant] Collection '{collectionName}' status: {createRes.StatusCode}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[Qdrant Error]: {ex.Message}");
                return;
            }

            Console.WriteLine("\n=== STEP 2: INDEXING ALL .CS FILES INTO QDRANT ===");
            string rootLabsDirectory = @"D:\azure-technology";

            if (!Directory.Exists(rootLabsDirectory))
            {
                Console.WriteLine($"[Error] Directory {rootLabsDirectory} not found!");
                return;
            }

            string[] filePaths = Directory.GetFiles(rootLabsDirectory, "*.cs", SearchOption.AllDirectories);
            int chunkIndex = 0;
            int linesPerChunk = 150;

            foreach (string filePath in filePaths)
            {
                if (filePath.Contains(@"\obj\") || filePath.Contains(@"\bin\")) continue;
                string fileName = Path.GetFileName(filePath);
                string[] lines = await File.ReadAllLinesAsync(filePath);

                string currentChunk = "";
                int lineCount = 0;

                foreach (var line in lines)
                {
                    currentChunk += line + "\n";
                    lineCount++;

                    if (lineCount >= linesPerChunk)
                    {
                        await IndexCurrentChunkAsync(currentChunk, fileName);
                        currentChunk = "";
                        lineCount = 0;
                    }
                }

                if (!string.IsNullOrWhiteSpace(currentChunk))
                {
                    await IndexCurrentChunkAsync(currentChunk, fileName);
                }
            }

            Console.WriteLine($"\n[Success] Total vectors indexed from all labs: {chunkIndex}");
            Console.WriteLine("Indexing complete. You can close this app and run SearchApp.");
            Console.ReadKey();

            async Task IndexCurrentChunkAsync(string chunkData, string fName)
            {
                if (string.IsNullOrWhiteSpace(chunkData)) return;
                chunkIndex++;
                string chunkWithMetadata = $"[File: {fName}]\n{chunkData.Trim()}";

                try
                {
                    var vector = await embeddingService.GenerateEmbeddingAsync(chunkWithMetadata);
                    var pointsBody = new
                    {
                        points = new[] {
                            new {
                                id = chunkIndex,
                                vector = vector.ToArray(),
                                payload = new { text = chunkWithMetadata }
                            }
                        }
                    };
                    var jsonContent = new StringContent(JsonSerializer.Serialize(pointsBody), Encoding.UTF8, "application/json");
                    var response = await qdrantHttpClient.PutAsync($"{qdrantEndpoint}/collections/{collectionName}/points", jsonContent);
                    if (response.IsSuccessStatusCode) Console.WriteLine($"Indexed chunk #{chunkIndex} from {fName}");
                }
                catch { }
                await Task.Delay(50);
            }
        }
    }
}