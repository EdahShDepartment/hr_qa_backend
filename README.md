# 營養指標自動化系統

## 🎯 專案簡介

在日常醫療工作中，營養科同仁每月需要統計大量指標資料以追蹤科室績效與支援決策。然而傳統做法主要依賴人工：

- 從 HIS 系統查詢、匯出資料
- 使用 Excel 手動計算與整理
- 編輯報表格式呈現

這些流程耗時又容易出現錯誤，增加同仁負擔。

本專案「營養指標自動化系統」，藉由平台幫助營養師統計各種營養指標。基於 Flask 的**營養指標自動化系統**，提供多種營養相關指標的資料處理功能。

---

## ✨ 主要功能

- **檔案上傳處理**：支援 Excel (.xlsx) 檔案上傳與自動處理
- **多指標類型**：目前支援四種不同的營養指標類型，後續可再新增
- **RESTful API**：標準化的 API 介面設計

## 安裝步驟

1. 下載或 clone 此專案

```bash
git clone [repository-url]
cd nutrition_dep_api
```

2. 安裝相依套件

```bash
pip install -r requirements.txt
```

3. 建立必要目錄

```bash
mkdir -p static/uploads
mkdir -p static/cancer_management_indicator
mkdir -p static/clinical_indicator
mkdir -p static/order_meal_statistics
mkdir -p static/rd_indicator
```

## 啟動方式

### 開發模式

```bash
python nutrition_indicators_app.py
```

預設會在 http://localhost:5002 運行。

### 正式部署

執行 `wsgi.py`：

```bash
python wsgi.py
```

---

## 🏗️ 專案結構

```
nutrition_dep_api/
├── __init__.py
├── nutrition_indicators_app.py     # 主應用程式、所有的服務、路由控制在這，未來要擴充api都是在這裡擴充(開發時可在這裡啟動 debug)
├── wsgi.py                         # 正式上線時啟動wsgi server
├── tools/                          # 指標處理工具
│   ├── cancer_management_indicator.py      #癌症管理指標
│   ├── clinical_indicator.py               #臨床指標
│   ├── order_meal_statistics.py            #訂餐統計指標
│   └── rd_indicator.py                     #CKD 指標
├── static/                         # 檔案上傳存放的位置
├── API_ENDPOINTS.md                # API 端點說明書
└── README.md                       # 專案說明文件
```

## 📡 API 端點概覽

| 端點                          | 方法 | 說明                         |
| ----------------------------- | ---- | ---------------------------- |
| `/`                           | GET  | 測試端點                     |
| `/upload/<type>`              | POST | 檔案上傳（支援多種指標類型） |
| `/download/<type>/<filename>` | GET  | 檔案下載                     |

### 支援的指標類型

- `order` - 訂餐統計指標
- `rd` - CKD 指標
- `cancer` - 癌症管理指標
- `clinical` - 臨床指標

## 🔧 配置說明

- 檔案上傳路徑：`./static/uploads/`
- 支援的檔案格式：`.xlsx`
- 檔案命名規則：自動加上時間戳記避免衝突

## 📚 詳細文件

- [API 端點完整說明](API_ENDPOINTS.md) - 詳細的 API 參數與回應格式

---

## 注意事項

- 本系統僅支援 `.xlsx` 格式的 Excel 檔案，上傳前請確認檔案格式正確。
- 上傳的 Excel 檔案，與要處理指標。

## 待改善 or 待開發

1. 未來**臨床績效**`clinical_indicator.py`其他分院或許會使用，再將他們的病房分群即可。注意不要更動目前本院的**分床邏輯**
2. 營養科之後有其他指標就在`tools/`新增其他指標處理的工具，在 ` nutrition_indicators_app.py` 的 `/upload/<type> `api 繼續延伸。

## 常見問題

- 指標選錯。