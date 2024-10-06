package com.xiaohub;


import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.splitter.DocumentByLineSplitter;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.input.Prompt;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.model.ollama.OllamaChatModel;
import dev.langchain4j.model.ollama.OllamaEmbeddingModel;
import dev.langchain4j.model.openai.OpenAiTokenizer;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import dev.langchain4j.store.embedding.chroma.ChromaEmbeddingStore;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import tech.amikos.chromadb.Client;

import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class LangChainMain {

    private static final Logger logger = LoggerFactory.getLogger(LoggerFactory.class);


    private static final String CHROMA_DB_DEFAULT_COLLECTION_NAME = "java-langChain-database-demo";
    private static final String OLLAMA_URL = "http://localhost:11434";
    private static final String CHROMA_URL = "http://localhost:8000";

    public static void main( String[] args ) {
        Document document = getDocument();

        splitFileContent(document);
    }

    private static Document getDocument(){
        logger.info("getDocument...");
        URL docUrl = LangChainMain.class.getClassLoader().getResource("笑话.txt");
        if (docUrl == null) {
            logger.error("未获取到文件");
            return null;
        }

        Document document = null;
        try {
            Path path = Paths.get(docUrl.toURI());
            document = FileSystemDocumentLoader.loadDocument(path);
        } catch (URISyntaxException e){
            logger.error("加载文件发生异常", e);
        }

        return document;
    }


    // ======================= 拆分文件内容=======================
    // 参数：分段大小（一个分段中最大包含多少个token）、重叠度（段与段之前重叠的token数）、分词器（将一段文本进行分词，得到token）
    private static void splitFileContent(Document document){
        DocumentByLineSplitter lineSplitter = new DocumentByLineSplitter(200, 0, new OpenAiTokenizer());
        List<TextSegment> segments = lineSplitter.split(document);
        logger.info("segment的数量是: {}", segments.size());

        // 查看分段后的信息
        segments.forEach(segment -> logger.info("========================segment: {}", segment.text()));

        embeddingModel(segments);
    }

    private static void embeddingModel(List<TextSegment> segments){
        // 文本向量化
        OllamaEmbeddingModel embeddingModel = OllamaEmbeddingModel.builder().baseUrl(OLLAMA_URL).modelName("qwen").build();

        // 向量库存储
        Client client = new Client(CHROMA_URL);

        // 创建向量数据库
        ChromaEmbeddingStore embeddingStore = ChromaEmbeddingStore.builder().baseUrl(CHROMA_URL).collectionName(CHROMA_DB_DEFAULT_COLLECTION_NAME).build();

        segments.forEach(segment -> {
            Embedding content = embeddingModel.embed(segment).content();
            embeddingStore.add(content, segment);
        });

        // 向量库检索
        String queryText = "北极熊";
        Embedding queryEmbedding = embeddingModel.embed(queryText).content();

        EmbeddingSearchRequest embeddingSearchRequest = EmbeddingSearchRequest.builder().queryEmbedding(queryEmbedding).maxResults(1).build();
        EmbeddingSearchResult<TextSegment> embeddedEmbeddingSearchResult  = embeddingStore.search(embeddingSearchRequest);
        List<EmbeddingMatch<TextSegment>> embeddingMatchList = embeddedEmbeddingSearchResult.matches();
        EmbeddingMatch<TextSegment> embeddingMatch = embeddingMatchList.get(0);
        TextSegment textSegment = embeddingMatch.embedded();
        logger.info("查询结果: {}", textSegment.text());

        buildPromptTemplate(textSegment);
    }

    private static void buildPromptTemplate(TextSegment textSegment) {
        //======================= 与LLM交互=======================
        PromptTemplate promptTemplate = PromptTemplate.from("基于如下信息用中文回答:\n" +
                "{{context}}\n" +
                "提问:\n" +
                "{{question}}");

        Map<String, Object> variables = new HashMap<>();
        // 以向量库检索到的结果作为LLM的信息输入
        variables.put("context", textSegment.text());
        variables.put("question", "北极熊干了什么");
        Prompt prompt = promptTemplate.apply(variables);

        // 连接大模型
        OllamaChatModel ollamaChatModel = OllamaChatModel.builder().baseUrl(OLLAMA_URL).modelName("qwen").build();

        UserMessage userMessage = prompt.toUserMessage();
        Response<AiMessage> aiMessageResponse = ollamaChatModel.generate(userMessage);
        AiMessage response = aiMessageResponse.content();

        logger.info("大模型回答: {}", response.text());

    }

}
