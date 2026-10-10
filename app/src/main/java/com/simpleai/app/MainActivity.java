package com.simpleai.app;

import android.app.Activity;
import android.Manifest;
import android.content.ActivityNotFoundException;
import android.content.pm.PackageManager;
import android.content.Intent;
import android.speech.RecognizerIntent;
import android.speech.tts.TextToSpeech;
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
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends Activity {
    private static final int VOICE_REQUEST_CODE = 42;
    private static final int AUDIO_PERMISSION_REQUEST_CODE = 43;
    private volatile String voiceLanguage = "en-US";
    private static final String MODEL_URL = "https://huggingface.co/litert-community/SmolLM-135M-Instruct/resolve/main/SmolLM-135M-Instruct_multi-prefill-seq_q8_ekv1280.task";
    private WebView webView;
    private TextToSpeech textToSpeech;
    private volatile boolean ttsReady = false;
    private LlmInference llmInference;
    private final ExecutorService modelExecutor = Executors.newSingleThreadExecutor();
    private volatile boolean modelLoading = false;
    private volatile boolean modelDownloading = false;
    private volatile String modelStatus = "Preparing your on-device AI model...";
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
        textToSpeech = new TextToSpeech(this, status -> {
            if (status == TextToSpeech.SUCCESS && textToSpeech != null) {
                ttsReady = true;
                textToSpeech.setLanguage(Locale.US);
            }
        });

        modelFile = new File(getFilesDir(), "simpleai-model.task");
        webView.loadUrl("file:///android_asset/index.html");
        if (modelFile.exists() && modelFile.length() > 100000000) {
            modelStatus = "Model found on this device. Loading...";
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
        modelStatus = "Downloading the AI model automatically (about 167 MB)...";
        sendJs("onLocalModelStatus", modelStatus);
        modelExecutor.execute(() -> {
            File temp = new File(getFilesDir(), "simpleai-model.tmp");
            HttpURLConnection connection = null;
            try {
                connection = (HttpURLConnection) new URL(MODEL_URL).openConnection();
                connection.setConnectTimeout(20000);
                connection.setReadTimeout(30000);
                connection.setInstanceFollowRedirects(true);
                connection.setRequestProperty("User-Agent", "SimpleAI-Android/0.0.5");
                connection.connect();
                int status = connection.getResponseCode();
                if (status < 200 || status >= 300) {
                    throw new IllegalStateException("The model server returned HTTP " + status + ".");
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
                                modelStatus = "Downloading model: " + percent + "%";
                                sendJs("onLocalModelStatus", modelStatus);
                            }
                        }
                    }
                    out.flush();
                }
                if (temp.length() < 100000000) {
                    throw new IllegalStateException("The model file is incomplete. Check your internet connection and try again.");
                }
                if (modelFile.exists()) modelFile.delete();
                if (!temp.renameTo(modelFile)) throw new IllegalStateException("Could not save the model to app storage.");
                runOnUiThread(() -> loadModel(modelFile));
            } catch (Exception e) {
                temp.delete();
                modelStatus = "Model download failed. Check your internet connection and try again. " + safeMessage(e);
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
        modelStatus = "Loading the model on this device. This may take a moment...";
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
                modelStatus = "Local AI is ready. Responses are generated on this device.";
                sendJs("onLocalModelStatus", modelStatus);
            } catch (Exception e) {
                modelStatus = "Could not load the model. Check that the .task file is compatible with MediaPipe. " + safeMessage(e);
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

    private String localeForLanguage(String language) {
        if (language == null) return "en-US";
        switch (language) {
            case "es": return "es-ES";
            case "fr": return "fr-FR";
            case "de": return "de-DE";
            case "fa": return "fa-IR";
            case "ar": return "ar";
            case "tr": return "tr-TR";
            case "hi": return "hi-IN";
            case "zh": return "zh-CN";
            case "ru": return "ru-RU";
            case "ja": return "ja-JP";
            default: return "en-US";
        }
    }

    private Locale localeObjectForLanguage(String language) {
        return Locale.forLanguageTag(localeForLanguage(language));
    }

    private void speakText(String text, String language) {
        runOnUiThread(() -> {
            if (textToSpeech == null || !ttsReady) {
                sendJs("onVoiceStatus", "Text-to-speech is still starting. Try again in a moment.");
                return;
            }
            Locale locale = localeObjectForLanguage(language);
            int availability = textToSpeech.setLanguage(locale);
            if (availability == TextToSpeech.LANG_MISSING_DATA || availability == TextToSpeech.LANG_NOT_SUPPORTED) {
                sendJs("onVoiceStatus", "A speech voice for this language is not installed on this device.");
                return;
            }
            textToSpeech.speak(text, TextToSpeech.QUEUE_FLUSH, null, "simple-ai-answer");
        });
    }

    private void beginVoiceInput() {
        try {
            Intent intent = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
            intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM);
            intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE, voiceLanguage);
            intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_PREFERENCE, voiceLanguage);
            intent.putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, true);
            intent.putExtra(RecognizerIntent.EXTRA_PROMPT, "Speak your message");
            startActivityForResult(intent, VOICE_REQUEST_CODE);
        } catch (ActivityNotFoundException e) {
            sendJs("onVoiceStatus", "Speech recognition is not available on this device.");
        } catch (Exception e) {
            sendJs("onVoiceStatus", "Could not start voice input: " + safeMessage(e));
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == AUDIO_PERMISSION_REQUEST_CODE) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                beginVoiceInput();
            } else {
                sendJs("onVoiceStatus", "Microphone permission was denied.");
            }
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode != VOICE_REQUEST_CODE) return;
        if (resultCode == RESULT_OK && data != null) {
            java.util.ArrayList<String> matches = data.getStringArrayListExtra(RecognizerIntent.EXTRA_RESULTS);
            if (matches != null && !matches.isEmpty()) {
                sendJs("onVoiceResult", matches.get(0));
                return;
            }
        }
        sendJs("onVoiceStatus", "No speech was recognized. Tap the microphone and try again.");
    }

    private class LocalModelBridge {
        @JavascriptInterface
        public void speakResponse(String text, String language) {
            if (text != null && !text.trim().isEmpty()) speakText(text, language);
        }

        @JavascriptInterface
        public void setVoiceLanguage(String language) {
            voiceLanguage = localeForLanguage(language);
        }

        @JavascriptInterface
        public void startVoiceInput(String language) {
            voiceLanguage = localeForLanguage(language);
            runOnUiThread(() -> {
                if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
                    requestPermissions(new String[]{Manifest.permission.RECORD_AUDIO}, AUDIO_PERMISSION_REQUEST_CODE);
                } else {
                    beginVoiceInput();
                }
            });
        }

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
                    sendJs("onLocalModelError", requestId, "The AI model is not ready yet. Please finish setup first.");
                    return;
                }
                try {
                    String formatted = "<|im_start|>system\nYou are Simple AI, a helpful multilingual assistant. Always answer in the same language as the user's latest message unless they explicitly ask for another language. Understand and respond in any language you can. Keep your tone clear, friendly, and respectful.\n<|im_end|>\n<|im_start|>user\n" + prompt
                            + "\n<|im_end|>\n<|im_start|>assistant\n";
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
        if (textToSpeech != null) {
            try { textToSpeech.stop(); textToSpeech.shutdown(); } catch (Exception ignored) {}
            textToSpeech = null;
        }
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
