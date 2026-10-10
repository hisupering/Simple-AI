package com.simpleai.app;

import android.app.Activity;
import android.content.Intent;
import android.database.Cursor;
import android.net.Uri;
import android.os.Bundle;
import android.provider.OpenableColumns;
import android.webkit.JavascriptInterface;
import android.webkit.WebChromeClient;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.graphics.Color;

import com.google.mediapipe.tasks.genai.llminference.LlmInference;

import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends Activity {
    private static final String MODEL_URL = "https://huggingface.co/litert-community/SmolLM-135M-Instruct/resolve/main/SmolLM-135M-Instruct_multi-prefill-seq_q8_ekv1280.task";
    private WebView webView;
    private LlmInference llmInference;
    private final ExecutorService modelExecutor = Executors.newSingleThreadExecutor();
    private volatile boolean modelLoading = false;
    private volatile boolean modelDownloading = false;
    private volatile String modelStatus = "در حال آماده‌سازی هوش مصنوعی محلی…";
    private volatile String activeRequestId = "";
    private File modelFile;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().setStatusBarColor(Color.rgb(255, 255, 255));
        getWindow().setNavigationBarColor(Color.rgb(255, 255, 255));
        getWindow().getDecorView().setSystemUiVisibility(
                android.view.View.SYSTEM_UI_FLAG_LIGHT_STATUS_BAR |
                android.view.View.SYSTEM_UI_FLAG_LIGHT_NAVIGATION_BAR);

        webView = new WebView(this);
        webView.setBackgroundColor(Color.rgb(255, 255, 255));
        webView.getSettings().setJavaScriptEnabled(true);
        webView.getSettings().setDomStorageEnabled(true);
        webView.setWebViewClient(new WebViewClient());
        webView.setWebChromeClient(new WebChromeClient());
        webView.addJavascriptInterface(new LocalModelBridge(), "SimpleLocalAI");
        setContentView(webView);

        modelFile = new File(getFilesDir(), "simpleai-model.task");
        webView.loadUrl("file:///android_asset/index.html");
        if (modelFile.exists() && modelFile.length() > 100000000) {
            modelStatus = "مدل روی گوشی پیدا شد؛ در حال آماده‌سازی…";
            loadModel(modelFile);
        } else {
            downloadModel();
        }
    }

    private void sendJs(String function, String... args) {
        if (webView == null) return;
        StringBuilder js = new StringBuilder("window.").append(function).append("(");
        for (int i = 0; i < args.length; i++) {
            if (i > 0) js.append(",");
            js.append(JSONObject.quote(args[i] == null ? "" : args[i]));
        }
        js.append(");");
        runOnUiThread(() -> {
            if (webView != null) webView.evaluateJavascript(js.toString(), null);
        });
    }

    private void downloadModel() {
        if (modelDownloading || modelLoading) return;
        modelDownloading = true;
        modelStatus = "در حال دریافت خودکار مدل هوش مصنوعی (حدود 167 مگابایت)…";
        sendJs("onLocalModelStatus", modelStatus);
        modelExecutor.execute(() -> {
            File temp = new File(getFilesDir(), "simpleai-model.tmp");
            HttpURLConnection connection = null;
            try {
                connection = (HttpURLConnection) new URL(MODEL_URL).openConnection();
                connection.setConnectTimeout(20000);
                connection.setReadTimeout(30000);
                connection.setInstanceFollowRedirects(true);
                connection.setRequestProperty("User-Agent", "SimpleAI-Android/0.0.3");
                connection.connect();
                int status = connection.getResponseCode();
                if (status < 200 || status >= 300) {
                    throw new IllegalStateException("سرور مدل پاسخ مناسب نداد (HTTP " + status + ").");
                }
                long total = connection.getContentLengthLong();
                try (InputStream in = connection.getInputStream();
                     FileOutputStream out = new FileOutputStream(temp)) {
                    byte[] buffer = new byte[256 * 1024];
                    int count;
                    long received = 0;
                    int lastPercent = -1;
                    while ((count = in.read(buffer)) != -1) {
                        received += count;
                        out.write(buffer, 0, count);
                        if (total > 0) {
                            int percent = (int) (received * 100 / total);
                            if (percent != lastPercent && percent % 5 == 0) {
                                lastPercent = percent;
                                modelStatus = "دریافت خودکار مدل: " + percent + "%";
                                sendJs("onLocalModelStatus", modelStatus);
                            }
                        }
                    }
                    out.flush();
                }
                if (temp.length() < 100000000) {
                    throw new IllegalStateException("فایل مدل ناقص است؛ اتصال اینترنت را بررسی کن و دوباره تلاش کن.");
                }
                if (modelFile.exists()) modelFile.delete();
                if (!temp.renameTo(modelFile)) throw new IllegalStateException("ذخیره مدل در حافظه برنامه ناموفق بود.");
                runOnUiThread(() -> loadModel(modelFile));
            } catch (Exception e) {
                temp.delete();
                modelStatus = "دریافت مدل انجام نشد. اینترنت را بررسی کن و برنامه را دوباره باز کن. " + safeMessage(e);
                sendJs("onLocalModelStatus", modelStatus);
            } finally {
                if (connection != null) connection.disconnect();
                modelDownloading = false;
            }
        });
    }

    private void loadModel(File file) {
        if (modelLoading) return;
        modelLoading = true;
        modelStatus = "در حال بارگذاری مدل روی گوشی… ممکن است کمی طول بکشد.";
        sendJs("onLocalModelStatus", modelStatus);
        modelExecutor.execute(() -> {
            try {
                LlmInference old = llmInference;
                llmInference = null;
                if (old != null) old.close();
                LlmInference.LlmInferenceOptions options =
                        LlmInference.LlmInferenceOptions.builder()
                                .setModelPath(file.getAbsolutePath())
                                .setMaxTokens(512)
.build();
                llmInference = LlmInference.createFromOptions(getApplicationContext(), options);
                modelStatus = "مدل محلی آماده است؛ پاسخ‌ها روی همین گوشی تولید می‌شوند.";
                sendJs("onLocalModelStatus", modelStatus);
            } catch (Exception e) {
                modelStatus = "بارگذاری مدل انجام نشد. مطمئن شو فایل، مدل سازگار با MediaPipe و پسوند .task است. " + safeMessage(e);
                sendJs("onLocalModelStatus", modelStatus);
            } finally {
                modelLoading = false;
            }
        });
    }

    private String safeMessage(Throwable error) {
        String message = error.getMessage();
        return message == null || message.trim().isEmpty() ? error.getClass().getSimpleName() : message;
    }

    private class LocalModelBridge {
        @JavascriptInterface
        public void selectModel() {
            downloadModel();
        }

        @JavascriptInterface
        public String getModelStatus() {
            return modelStatus;
        }

        @JavascriptInterface
        public boolean isModelReady() {
            return llmInference != null && !modelLoading;
        }

        @JavascriptInterface
        public boolean isModelDownloading() {
            return modelDownloading;
        }

        @JavascriptInterface
        public void generate(String prompt, String requestId) {
            modelExecutor.execute(() -> {
                LlmInference engine = llmInference;
                activeRequestId = requestId;
                if (engine == null) {
                    sendJs("onLocalModelError", requestId, "هنوز مدلی بارگذاری نشده. اول فایل مدل .task را انتخاب کن.");
                    return;
                }
                try {
                    String formatted = "<|im_start|>user\\n" + prompt
                            + "<|im_end|>\\n<|im_start|>assistant\\n";
                    String response = engine.generateResponse(formatted);
                    sendJs("onLocalModelChunk", requestId, response, "true");
                } catch (Exception e) {
                    sendJs("onLocalModelError", requestId, safeMessage(e));
                }
            });
        }
    }

    @Override
    protected void onDestroy() {
        if (llmInference != null) {
            try { llmInference.close(); } catch (Exception ignored) {}
            llmInference = null;
        }
        modelExecutor.shutdown();
        if (webView != null) {
            webView.removeJavascriptInterface("SimpleLocalAI");
            webView.destroy();
            webView = null;
        }
        super.onDestroy();
    }

    @Override
    public void onBackPressed() {
        if (webView != null && webView.canGoBack()) webView.goBack();
        else super.onBackPressed();
    }
}
