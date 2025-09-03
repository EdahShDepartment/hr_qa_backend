# 營養指標系統 API 說明書

本文件說明 nutrition_indicators_app.py 提供的 api-endpoints、parameters 與 response 格式。

---

- **prefix**:`nutrition_indicators-api`
  在`app.py` _app.config['APPLICATION_ROOT']_ 調整

## 1. 首頁測試端點

- **URL**：`/`
- **方法**：GET
- **說明**：測試 API 是否正常運作。
- **回應範例**：

```json
{
  "status": 200,
  "message": "nutrition_indicators-api|Test endpoint is working",
  "result": [],
  "success": true
}
```

---

## 2. 檔案上傳端點

- **URL**：`/upload/<type>`
- **方法**：POST
- **說明**：上傳 Excel 檔案（僅限 .xlsx），並根據 type 進行不同的指標處理。
- **路徑參數**：
  - `type`：指標類型，支援 `order`、`rd`、`cancer`、`clinical`
- **表單參數**：
  - `file`：要上傳的 .xlsx 檔案
  - `month`：僅當 type 為 `order` 時需要，指定月份（整數）
- **回應**：
  - **成功**（201）：
    ```json
    {
      "status": 201,
      "message": "檔案上傳成功",
      "filename": "xxx_yyyymmddHHMMSS_.xlsx",
      "success": true
    }
    ```
  - **失敗**（400/500）：
    - 檔案格式錯誤、缺少檔案、未知 type、指標處理錯誤等，會回傳對應錯誤訊息。
    - 例如：
      ```json
      {
        "status": 400,
        "message": "只可以上傳.xlsx",
        "success": false
      }
      ```
      ```json
      {
        "status": 400,
        "message": "未知的指標類型",
        "success": false
      }
      ```
      ```json
      {
        "status": 400,
        "message": "請確認檔案及指標是否選錯",
        "success": false
      }
      ```
      ```json
      {
        "status": 400,
        "message": "Stage0未刪除，請檢查",
        "success": false
      }
      ```

---

## 3. 檔案下載端點

- **URL**：`/download/<type>/<filename>`
- **方法**：GET
- **說明**：下載已上傳並處理過的檔案。
- **路徑參數**：
  - `type`：指標類型，支援 `order`、`rd`、`cancer`、`clinical`
  - `filename`：檔案名稱（含副檔名）
- **回應**：
  - **成功**：下載檔案
  - **失敗**（404）：
    ```json
    {
      "status": 404,
      "error": "檔案不存在",
      "success": false
    }
    ```
- **注意**:此端點直接回傳檔案內容，不回傳 JSON 結構

---

## 4. 支援的指標類型（type）

- `order`：訂餐統計指標（需額外提供 `month` 參數）
- `rd`：CKD 指標
- `cancer`：癌症管理指標
- `clinical`：臨床指標

---

## 5. 其他說明

- 上傳的檔案會自動加上 timestamp 以避免檔名衝突。
- 僅支援 `.xlsx` 格式。