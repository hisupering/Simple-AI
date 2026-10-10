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
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends Activity {
    private static final int PICK_MODEL_REQUEST = 4103;
    private WebView webView;
    private LlmInference llmInference;
    private final ExecutorService modelExecutor = Executors.newSingleThreadExecutor();
    private volatile boolean modelLoading = false;
    private volatile String modelStatus = "برای شروع، فایل مدل .task را انتخاب کن.";
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
        if (modelFile.exists() && modelFile.length() > 0) {
            modelStatus = "فایل مدل ذخیره شده است؛ در حال آماده‌سازی…";
            loadModel(modelFile);
        }
        webView.loadUrl("file:///android_asset/index.html");
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

    private void pickModel() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("*/*");
        startActivityForResult(intent, PICK_MODEL_REQUEST);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode != PICK_MODEL_REQUEST || resultCode != RESULT_OK || data == null || data.getData() == null) return;
        Uri uri = data.getData();
        modelStatus = "در حال کپی فایل مدل به حافظه برنامه…";
        sendJs("onLocalModelStatus", modelStatus);
        modelExecutor.execute(() -> {
            File temp = new File(getFilesDir(), "simpleai-model.tmp");
            try (InputStream in = getContentResolver().openInputStream(uri);
                 FileOutputStream out = new FileOutputStream(temp)) {
                if (in == null) throw new IllegalStateException("فایل قابل خواندن نیست.");
                byte[] buffer = new byte[1024 * 1024];
                int count;
                while ((count = in.read(buffer)) != -1) out.write(buffer, 0, count);
                out.flush();
                if (!temp.renameTo(modelFile)) {
                    if (modelFile.exists()) modelFile.delete();
                    if (!temp.renameTo(modelFile)) throw new IllegalStateException("کپی فایل مدل کامل نشد.");
                }
                runOnUiThread(() -> loadModel(modelFile));
            } catch (Exception e) {
                temp.delete();
                modelStatus = "خواندن فایل مدل ناموفق بود: " + safeMessage(e);
                sendJs("onLocalModelStatus", modelStatus);
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
                                .setResultListener((partialResult, done) -> {
                                    // Partial chunks are streamed to the chat UI.
                                    sendJs("onLocalModelChunk", activeRequestId, partialResult, done ? "true" : "false");
                                })
                                .setErrorListener(error -> {
                                    String message = "خطای مدل: " + safeMessage(error);
                                    modelStatus = message;
                                    sendJs("onLocalModelStatus", message);
                                    sendJs("onLocalModelError", activeRequestId, message);
                                })
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
            runOnUiThread(() -> pickModel());
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
        public void generate(String prompt, String requestId) {
            modelExecutor.execute(() -> {
                LlmInference engine = llmInference;
                activeRequestId = requestId;
                if (engine == null) {
                    sendJs("onLocalModelError", requestId, "هنوز مدلی بارگذاری نشده. اول فایل مدل .task را انتخاب کن.");
                    return;
                }
                try {
                    String formatted = "<start_of_turn>user\n" + prompt
                            + "\n<end_of_turn>\n<start_of_turn>model\n";
                    engine.generateResponseAsync(formatted);
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
