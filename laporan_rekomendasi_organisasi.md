# 📋 Laporan Pengembangan Fitur Rekomendasi Organisasi
## Portal Open Data — Halaman `data-saya/catalog-create`

**Tanggal Pengerjaan:** 14 April 2026  
**Waktu:** 11:21 – 12:46 WIB  
**Proyek:** `portal-open-data` (Laravel 13 + React/Inertia.js)  
**ML Service:** `ML Auto-Tagging Dashboard` (FastAPI + Scikit-Learn)

---

## 1. Latar Belakang & Tujuan

Portal Open Data Kementerian memiliki halaman **Buat Posting Data** (`/data-saya/catalog-create`) yang digunakan oleh pengguna internal untuk membuat entri katalog dataset baru.

Form tersebut mengharuskan pengguna untuk memilih **Unit Kerja Eselon 1** dan **Unit Kerja Eselon 2** secara manual. Proses ini memakan waktu dan berisiko salah pilih, terutama bagi pengguna yang belum familiar dengan struktur organisasi.

**Tujuan pengembangan ini** adalah:
> Menampilkan **rekomendasi organisasi secara otomatis** berdasarkan nama dataset yang diketikkan, menggunakan model Machine Learning (ML) yang sudah ada di sistem, dan menampilkan hasilnya dalam bentuk **kartu interaktif** (card) di bawah select-box Unit Kerja Eselon 2, sehingga pengguna dapat langsung mengklik kartu untuk mengisi select-box secara otomatis.

---

## 2. Analisis Sistem yang Ada

### 2.1 Halaman `catalog-create`

File: [resources/js/Pages/Admin/Catalog/Create.tsx](file:///home/ifhalfaizi/portal_open_data/portal-open-data/resources/js/Pages/Admin/Catalog/Create.tsx)

Form ini memiliki 4 field utama:
- **Nama Kumpulan Data** (`group_name`) — text input
- **Nama Data** (`name`) — text input (nama dataset)
- **Unit Kerja Eselon 1** (`prod_data_es_1`) — Select2 dropdown
- **Unit Kerja Eselon 2** (`prod_data_es_2`) — Select2 dropdown (bergantung pada pilihan eselon 1)

Eselon 1 dimuat dari endpoint `/data-saya/fetch_catalog_org`.  
Eselon 2 dimuat dari endpoint `/data-saya/fetch_catalog_sub_org/{id}` berdasarkan pilihan eselon 1.

### 2.2 Controller Utama

File: [app/Http/Controllers/AdminCatalogController.php](file:///home/ifhalfaizi/portal_open_data/portal-open-data/app/Http/Controllers/AdminCatalogController.php)

method-method yang relevan:
- [create()](file:///home/ifhalfaizi/portal_open_data/portal-open-data/app/Http/Controllers/AdminCatalogController.php#89-125) — Render halaman buat posting data
- [store()](file:///home/ifhalfaizi/portal_open_data/portal-open-data/app/Http/Controllers/AdminCatalogController.php#126-164) — Validasi & simpan posting data baru
- [fetchOrg()](file:///home/ifhalfaizi/portal_open_data/portal-open-data/app/Http/Controllers/AdminCatalogController.php#429-435) — Ambil daftar eselon 1 (JSON)
- [fetchSubOrg()](file:///home/ifhalfaizi/portal_open_data/portal-open-data/app/Http/Controllers/AdminCatalogController.php#436-448) — Ambil daftar eselon 2 berdasarkan parent eselon 1 (JSON)

### 2.3 Model ML yang Tersedia

Proyek **ML Auto-Tagging Dashboard** di direktori `/home/ifhalfaizi/ML Auto-Tagging Dashboard/` mengandung:
- Model Scikit-Learn (`.joblib`) untuk klasifikasi `group` dan `organization`
- FastAPI microservice di direktori `ml_app/`
- Fungsi `predict_file_topk()` yang mengembalikan top-K prediksi beserta **confidence score**

Format response ML:
```json
{
  "filename": "jumlah-alat-telekomunikasi.csv",
  "pred_organization": "direktorat-jenderal-sumber-daya-dan-perangkat-pos-dan-informatika",
  "org_topk": [
    { "rank": 1, "label": "direktorat-jenderal-...", "confidence": 91.7 },
    { "rank": 2, "label": "direktorat-jenderal-informasi-...", "confidence": 2.5 },
    { "rank": 3, "label": "badan-penelitian-...", "confidence": 2.0 }
  ]
}
```

> **Catatan penting:** Label dari ML dikembalikan dalam format **slug** (huruf kecil, tanda `-` sebagai pemisah kata), bukan nama asli dengan spasi seperti di database.

---

## 3. Arsitektur Solusi

```
┌───────────────────────────────────────────────┐
│           Browser (React/Inertia.js)           │
│  ┌─────────────────────────────────────────┐  │
│  │ CatalogCreate.tsx                       │  │
│  │  - FormInput: Nama Data (debounce 800ms)│  │
│  │  - Select2: Eselon 1                    │  │
│  │  - Select2: Eselon 2                    │  │
│  │  - [💡 Kartu Rekomendasi] ← NEW         │  │
│  └─────────────────┬───────────────────────┘  │
└────────────────────┼──────────────────────────┘
                     │ GET /data-saya/recommend_org?name=...
                     ▼
┌───────────────────────────────────────────────┐
│           Laravel (AdminCatalogController)     │
│  recommendOrg()                                │
│    ├─ POST ML_SERVICE_URL/predict              │
│    │   ├─ Berhasil → ambil org_topk            │
│    │   └─ Gagal/timeout → DB fallback          │
│    └─ Enrich setiap item:                      │
│        ├─ org_id (ID eselon 2 di DB)           │
│        └─ org_parent_id (ID eselon 1 di DB)   │
└────────────────────┬──────────────────────────┘
                     │ POST /predict
                     ▼
┌───────────────────────────────────────────────┐
│     FastAPI ML Service (port 8002)             │
│     uvicorn main:app --host 0.0.0.0 --port 8002│
│     classifier.predict_file_topk(filename)     │
└───────────────────────────────────────────────┘
```

---

## 4. Implementasi Detail

### 4.1 Tahap 1 — Menambahkan Route Baru

**File:** `routes/web.php`

Di dalam middleware group `data-saya` (auth guard), ditambahkan satu route baru:

```php
Route::get('/recommend_org', [AdminCatalogController::class, 'recommendOrg'])
    ->name('catalog.recommend.org');
```

**URL lengkap:** `GET /data-saya/recommend_org?name={nama_dataset}`

---

### 4.2 Tahap 2 — Method `recommendOrg` di Controller

**File:** `app/Http/Controllers/AdminCatalogController.php`

Method baru `recommendOrg(Request $request)` ditambahkan setelah method `fetchSubOrg`. Logika lengkap:

```php
public function recommendOrg(Request $request): \Illuminate\Http\JsonResponse
{
    $name = (string) $request->query('name', '');

    if (trim($name) === '') {
        return response()->json([]);
    }

    $rawResults = [];

    // --- STEP 1: Coba FastAPI ML Service ---
    $mlUrl = config('constants.ML_SERVICE_URL', env('ML_SERVICE_URL', ''));
    if (!empty($mlUrl)) {
        try {
            $mlResponse = Http::timeout(3)
                ->post(rtrim($mlUrl, '/') . '/predict', ['filename' => $name]);

            if ($mlResponse->successful()) {
                $orgTopk = $mlResponse->json()['org_topk'] ?? [];
                foreach ($orgTopk as $item) {
                    $rawResults[] = [
                        'rank'       => $item['rank'],
                        'label'      => $item['label'],
                        'confidence' => $item['confidence'] ?? null,
                        'org_id'     => null,
                    ];
                }
            }
        } catch (\Exception $e) {
            Log::info('ML Service timeout, fallback ke DB');
        }
    }

    // --- STEP 2: Fallback ke DB jika ML kosong ---
    if (empty($rawResults)) {
        $keywords = preg_split('/\s+/', strtolower(trim($name)));
        $query = Organization::where('is_active', true)->whereNotNull('parent_code');
        foreach ($keywords as $kw) {
            if (strlen($kw) >= 3) {
                $query->where(fn($q) => $q->whereRaw('LOWER(name) LIKE ?', ["%$kw%"]));
            }
        }
        $dbResults = $query->select('id', 'name')->limit(3)->get();
        // ... (map ke rawResults)
    }

    // --- STEP 3: Enrich dengan org_id dan org_parent_id dari DB ---
    $formatted = [];
    foreach ($rawResults as $item) {
        $orgId = $item['org_id'];

        // Konversi slug → spasi sebelum LIKE search
        if ($orgId === null) {
            $searchLabel = str_replace('-', ' ', strtolower(trim($item['label'])));
            $org = Organization::where('is_active', true)
                ->whereRaw('LOWER(name) LIKE ?', ["%$searchLabel%"])
                ->first();
            $orgId = $org?->id;
        }

        // Cari parent (eselon 1)
        $orgParentId = null;
        if ($orgId !== null) {
            $orgRecord = Organization::find($orgId);
            $parent = Organization::where('code', $orgRecord->parent_code)
                ->where('type', 'eselon 1')->first();
            $orgParentId = $parent?->id;
        }

        $formatted[] = [
            'rank'          => $item['rank'],
            'label'         => $item['label'],
            'confidence'    => $item['confidence'],
            'org_id'        => $orgId,        // ID eselon 2
            'org_parent_id' => $orgParentId,  // ID eselon 1
        ];
    }

    return response()->json($formatted);
}
```

**Contoh response API:**
```json
[
  {
    "rank": 1,
    "label": "direktorat-jenderal-sumber-daya-dan-perangkat-pos-dan-informatika",
    "confidence": 91.7,
    "org_id": 42,
    "org_parent_id": 7
  },
  {
    "rank": 2,
    "label": "direktorat-jenderal-informasi-dan-komunikasi-publik",
    "confidence": 2.5,
    "org_id": 38,
    "org_parent_id": 7
  }
]
```

---

### 4.3 Tahap 3 — Konfigurasi `ML_SERVICE_URL`

**File:** `config/constants.php`

```php
'ML_SERVICE_URL' => env('ML_SERVICE_URL', ''),
```

**File:** `.env`

```env
ML_SERVICE_URL=http://127.0.0.1:8002
```

Kemudian clear config cache:
```bash
php artisan config:clear
```

---

### 4.4 Tahap 4 — Frontend `Create.tsx`

**File:** `resources/js/Pages/Admin/Catalog/Create.tsx`

#### 4.4.1 Type Definition Baru

```tsx
interface OrgRecommendation {
    rank: number;
    label: string;
    confidence: number | null;
    org_id: number | null;
    org_parent_id: number | null;
}
```

#### 4.4.2 State dan Ref Baru

```tsx
const [orgRecommendations, setOrgRecommendations] = useState<OrgRecommendation[]>([]);
const [isLoadingRecommendations, setIsLoadingRecommendations] = useState(false);
const pendingEs2Ref = useRef<string | null>(null);   // pending eselon 2 ID
const debounceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
```

#### 4.4.3 Debounced Fetch Rekomendasi

Saat `formData.name` berubah, fetch dijalankan setelah **800ms delay** (debounce):

```tsx
useEffect(() => {
    const nameValue = formData.name.trim();
    if (debounceTimerRef.current) clearTimeout(debounceTimerRef.current);
    if (nameValue.length < 3) { setOrgRecommendations([]); return; }

    debounceTimerRef.current = setTimeout(() => {
        setIsLoadingRecommendations(true);
        fetch(`/data-saya/recommend_org?name=${encodeURIComponent(nameValue)}`)
            .then(res => res.json())
            .then((recs: OrgRecommendation[]) => setOrgRecommendations(recs))
            .catch(() => setOrgRecommendations([]))
            .finally(() => setIsLoadingRecommendations(false));
    }, 800);
}, [formData.name]);
```

#### 4.4.4 Pending Eselon 2 Mechanism

Masalah timing: eselon 2 options baru tersedia **setelah** eselon 1 dipilih (async fetch). Solusinya menggunakan `pendingEs2Ref`:

```tsx
useEffect(() => {
    if (formData.prod_data_es_1) {
        fetch(`/data-saya/fetch_catalog_sub_org/${formData.prod_data_es_1}`)
            .then(res => res.json())
            .then(subData => {
                const opts = subData.map(...);
                setEselon2Options(opts);

                // Apply pending eselon 2 jika ada
                if (pendingEs2Ref.current !== null) {
                    const pendingId = pendingEs2Ref.current;
                    pendingEs2Ref.current = null;
                    const found = opts.find(o => String(o.value) === String(pendingId));
                    if (found) {
                        setFormData(prev => ({ ...prev, prod_data_es_2: String(found.value) }));
                    }
                }
            });
    }
}, [formData.prod_data_es_1]);
```

#### 4.4.5 Algoritma Pemilihan Kartu (3-Tier Matching)

```tsx
const handleSelectRecommendation = (rec: OrgRecommendation) => {
    const normalize = (s: string) =>
        s.replace(/-/g, ' ').replace(/\s+/g, ' ').trim().toLowerCase();

    const labelNorm = normalize(rec.label);

    // --- TIER 1: Gunakan org_parent_id dari backend (paling akurat) ---
    if (rec.org_parent_id) {
        pendingEs2Ref.current = rec.org_id ? String(rec.org_id) : null;
        setFormData(prev => ({ ...prev, prod_data_es_1: String(rec.org_parent_id), prod_data_es_2: '' }));
        return;
    }

    // --- TIER 2 & 3: Frontend fuzzy match terhadap eselon1Options ---
    let bestMatch: SelectOption | null = null;
    let bestScore = 0;

    for (const opt of eselon1Options) {
        const optNorm = normalize(String(opt.label));

        // Exact match
        if (optNorm === labelNorm) { bestMatch = opt; break; }

        // Contains match
        if (optNorm.includes(labelNorm) || labelNorm.includes(optNorm)) {
            const score = Math.min(optNorm.length, labelNorm.length);
            if (score > bestScore) { bestScore = score; bestMatch = opt; }
            continue;
        }

        // Word overlap scoring (kata ≥ 4 huruf)
        const labelWords = new Set(labelNorm.split(' ').filter(w => w.length > 3));
        const optWords = optNorm.split(' ').filter(w => w.length > 3);
        const overlap = optWords.filter(w => labelWords.has(w)).length;
        if (overlap > 0) {
            const score = overlap * 10 + Math.min(optNorm.length, labelNorm.length);
            if (score > bestScore) { bestScore = score; bestMatch = opt; }
        }
    }

    if (bestMatch) {
        pendingEs2Ref.current = rec.org_id ? String(rec.org_id) : null;
        setFormData(prev => ({ ...prev, prod_data_es_1: String(bestMatch!.value), prod_data_es_2: '' }));
    }
};
```

#### 4.4.6 Tampilan UI Kartu Rekomendasi

Di bawah select-box eselon 2, ditampilkan section rekomendasi:

```tsx
{(isLoadingRecommendations || orgRecommendations.length > 0) && (
    <div className="mb-4">
        <div className="d-flex align-items-center mb-2 gap-2">
            <span>💡</span>
            <small className="fw-semibold text-muted">Rekomendasi Organisasi</small>
            {isLoadingRecommendations && <div className="spinner-border spinner-border-sm" />}
        </div>

        <div className="d-flex flex-wrap gap-2">
            {orgRecommendations.map(rec => (
                <div key={rec.rank} role="button" onClick={() => handleSelectRecommendation(rec)}
                    style={{ border: '1.5px solid #dee2e6', borderRadius: '10px', ... }}>

                    {/* Badge rank dan confidence % */}
                    <div className="d-flex justify-content-between">
                        <span>#{rec.rank}</span>
                        {rec.confidence !== null && <span>{rec.confidence}%</span>}
                    </div>

                    {/* Nama organisasi */}
                    <div>{rec.label}</div>

                    {/* Progress bar confidence */}
                    <div style={{ height: '4px', background: confidenceColor(rec.confidence) }} />

                    <div>↑ Klik untuk pilih</div>
                </div>
            ))}
        </div>
    </div>
)}
```

**Warna confidence bar:**
| Confidence | Warna |
|---|---|
| ≥ 70% | 🟢 Hijau (`#198754`) |
| 40–69% | 🟠 Oranye (`#fd7e14`) |
| < 40% | 🔴 Merah (`#dc3545`) |
| `null` | 🔘 Abu-abu (`#6c757d`) |

---

## 5. Troubleshooting yang Dilakukan

### 5.1 Port ML Service Sudah Dipakai (Error 98)

**Masalah:** `uvicorn` gagal start di port 8001 — `[Errno 98] Address already in use`

**Solusi:** Pindah ke port 8002
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8002
```
Dan update `.env`:
```env
ML_SERVICE_URL=http://127.0.0.1:8002
```

---

### 5.2 Klik Kartu Tidak Mengisi Select-Box

**Root cause #1:** Backend tidak mengembalikan `org_id`/`org_parent_id` — hanya `label` string dari ML.

**Perbaikan:** Backend di-enrich untuk lookup `org_id` dan `org_parent_id` dari tabel `organization` di DB.

---

### 5.3 DB Lookup Gagal — Slug vs Spasi

**Root cause #2:** Label ML menggunakan format **slug** dengan tanda `-`:
```
direktorat-jenderal-sumber-daya-dan-perangkat-pos-dan-informatika
```
Sedangkan DB menyimpan dengan **spasi**:
```
DIREKTORAT JENDERAL SUMBER DAYA DAN PERANGKAT POS DAN INFORMATIKA
```

Query `LIKE '%direktorat-jenderal-...%'` tidak cocok karena ada `-` vs ` `.

**Perbaikan (backend):**
```php
$searchLabel = str_replace('-', ' ', strtolower(trim($item['label'])));
Organization::whereRaw('LOWER(name) LIKE ?', ['%' . $searchLabel . '%'])->first();
```

---

### 5.4 Frontend Matching Sebagai Solusi Final

Karena DB lookup masih kadang menghasilkan `null` (misalnya data tidak sinkron), ditambahkan **frontend fuzzy matching** sebagai fallback primer:

```
"direktorat-jenderal-sumber-daya-dan-perangkat-pos-dan-informatika"
         ↓ normalize: replace(-→ ) + lowercase
"direktorat jenderal sumber daya dan perangkat pos dan informatika"
         ↓ dibandingkan dengan eselon1Options (word overlap scoring)
"DIREKTORAT JENDERAL SUMBER DAYA DAN PERANGKAT POS DAN INFORMATIKA" ← MATCH ✅
```

---

## 6. File yang Dimodifikasi

| No | File | Jenis Perubahan |
|----|------|-----------------|
| 1 | `routes/web.php` | Tambah route `GET /recommend_org` |
| 2 | `app/Http/Controllers/AdminCatalogController.php` | Tambah method `recommendOrg()` |
| 3 | `config/constants.php` | Tambah key `ML_SERVICE_URL` |
| 4 | `.env` | Tambah `ML_SERVICE_URL=http://127.0.0.1:8002` |
| 5 | `resources/js/Pages/Admin/Catalog/Create.tsx` | Tambah UI kartu + logic pemilihan |

---

## 7. Alur Kerja Lengkap Fitur

```
User buka halaman /data-saya/catalog-create
      │
      ▼
User mengetik Nama Data (≥ 3 karakter)
      │
      ▼ debounce 800ms
      │
      ▼
Frontend fetch:
GET /data-saya/recommend_org?name={nama_dataset}
      │
      ▼
AdminCatalogController::recommendOrg()
      ├─ POST http://127.0.0.1:8002/predict  ← FastAPI ML
      │       └─ Response: org_topk[{rank, label(slug), confidence}]
      │
      ├─ Enrich setiap item:
      │   ├─ str_replace('-', ' ', label) → LIKE search → org_id (eselon 2)
      │   └─ Cari parent_code → org_parent_id (eselon 1)
      │
      └─ JSON Response: [{rank, label, confidence, org_id, org_parent_id}]
      │
      ▼
Frontend tampilkan 3 kartu rekomendasi:
  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
  │ #1      91.7%│  │ #2       2.5%│  │ #3         2%│
  │ direktorat.. │  │ direktorat.. │  │ badan-peneli │
  │ ████████░░░ │  │ █░░░░░░░░░░ │  │ █░░░░░░░░░░ │
  │ ↑ Klik pilih│  │ ↑ Klik pilih│  │ ↑ Klik pilih│
  └──────────────┘  └──────────────┘  └──────────────┘
      │
      ▼ User klik kartu #1
      │
handleSelectRecommendation(rec):
  TIER 1: org_parent_id tersedia? → set prod_data_es_1 = org_parent_id
  TIER 2: slug normalize → fuzzy match eselon1Options
  TIER 3: word overlap scoring
      │
      ▼
pendingEs2Ref = org_id (eselon 2 ID)
setFormData({ prod_data_es_1: es1Id, prod_data_es_2: '' })
      │
      ▼ useEffect dipicu (prod_data_es_1 berubah)
      │
fetch /data-saya/fetch_catalog_sub_org/{es1Id}
      │
      ▼
setEselon2Options(opts)
Apply pendingEs2Ref → setFormData({ prod_data_es_2: pending_id })
      │
      ▼
✅ Eselon 1 dan Eselon 2 terisi otomatis!
```

---

## 8. Konfigurasi Lingkungan (Environment)

### Laravel (portal-open-data)
```bash
# Jalankan backend
php artisan serve           # http://localhost:8000

# Jalankan frontend bundler
npm run dev
```

### FastAPI ML Service
```bash
cd /home/ifhalfaizi/"ML Auto-Tagging Dashboard"
source venv/bin/activate
cd ml_service
uvicorn main:app --reload --host 0.0.0.0 --port 8002
```

### Variabel Environment Penting
```env
# .env (portal-open-data)
APP_URL=http://localhost:8000
ML_SERVICE_URL=http://127.0.0.1:8002
DB_CONNECTION=pgsql
DB_DATABASE=portal_open_data
```

---

## 9. Desain UI Kartu Rekomendasi

### Screenshot Hasil Akhir

Tampilan kartu rekomendasi di bawah select-box eselon 2:

```
┌─────────────────────────────────────────────────────────────────┐
│ 💡 Rekomendasi Organisasi                                        │
│                                                                  │
│ ┌───────────────────────┐ ┌──────────────────────┐ ┌──────────┐│
│ │ #1              91.7% │ │ #2               2.5% │ │ #3    2% ││
│ │ direktorat-jenderal-  │ │ direktorat-jenderal-  │ │ badan-  ││
│ │ sumber-daya-dan-      │ │ informasi-dan-        │ │ penelit-││
│ │ perangkat-pos-dan-    │ │ komunikasi-publik     │ │ ian-... ││
│ │ informatika           │ │                       │ │         ││
│ │ ████████████░░░░░    │ │ █░░░░░░░░░░░░░░░░    │ │ ░░░░░░░ ││
│ │ ↑ Klik untuk pilih   │ │ ↑ Klik untuk pilih   │ │ ↑ Klik  ││
│ └───────────────────────┘ └──────────────────────┘ └──────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### Spesifikasi Visual
- **Border radius:** 10px
- **Shadow:** `0 1px 4px rgba(0,0,0,0.07)`
- **Hover:** border berubah biru `#0d6efd`, shadow lebih kuat
- **Badge rank:** background `#e9ecef`, font bold
- **Confidence bar:** height 4px, warna sesuai nilai (hijau/oranye/merah)
- **Teks "Klik untuk pilih":** biru `#0d6efd`, font-weight 500

---

## 10. Ringkasan Teknis

| Aspek | Detail |
|-------|--------|
| **Backend** | Laravel 13 (PHP 8.4) |
| **Frontend** | React 18 + TypeScript + Inertia.js |
| **ML Service** | FastAPI + Uvicorn + Scikit-Learn |
| **Komunikasi** | HTTP POST (Laravel → FastAPI), timeout 3 detik |
| **Debounce** | 800ms setelah user berhenti mengetik |
| **Minimum karakter** | 3 karakter untuk trigger fetch |
| **Jumlah rekomendasi** | Top-3 (rank 1, 2, 3) |
| **Fallback strategy** | DB keyword search jika ML tidak tersedia |
| **Matching strategy** | 3-tier: ID exact → slug normalize → word overlap |
| **Pending mechanism** | `useRef` untuk apply eselon 2 setelah opsi dimuat |

---

## 11. Catatan & Rekomendasi Ke Depan

1. **Training Data ML:** Label organisasi dalam training data menggunakan format slug. Pertimbangkan untuk menyimpan mapping slug → ID di database agar lookup lebih cepat dan akurat.

2. **Cache Rekomendasi:** Pertimbangkan menambahkan cache di backend untuk nama yang sering dicari, mengurangi beban ke ML service.

3. **Feedback Loop:** Tambahkan logging ketika user memilih rekomendasi (kartu mana yang dipilih) untuk meningkatkan akurasi model di masa depan.

4. **Fallback UI:** Jika tidak ada rekomendasi sama sekali, tampilkan pesan informatif daripada section kosong.

5. **Mobile Responsive:** Kartu sudah menggunakan `flex-wrap`, namun perlu diuji di layar mobile kecil.

---

*Laporan dibuat otomatis pada 14 April 2026 — Portal Open Data Kementerian*

---

## 12. Perbaikan Bug — Sesi 14 April 2026 (Siang)

### 12.1 Dataset Tidak Muncul di Halaman Publik

**File:** `app/Http/Controllers/OpenDataController.php`

**Root Cause:** Query di method `searchDatasetsFromDatabase()` memiliki filter `->whereNotNull('ckan_id')->where('ckan_id', '!=', '')` yang menyebabkan semua catalog lokal tidak muncul, karena 3 data yang ada di database tidak memiliki `ckan_id`.

**Perbaikan:** Hapus kedua filter tersebut agar catalog lokal tanpa `ckan_id` juga tampil.

```php
// SEBELUM
$query = Catalog::with(['files', 'organizationEs1', 'organizationEs2', 'plan'])
    ->where('is_deleted', 0)
    ->whereNotNull('ckan_id')       // ← dihapus
    ->where('ckan_id', '!=', '')    // ← dihapus
    ->orderBy('created_at', 'desc');

// SESUDAH
$query = Catalog::with(['files', 'organizationEs1', 'organizationEs2', 'plan'])
    ->where('is_deleted', 0)
    ->orderBy('created_at', 'desc');
```

---

### 12.2 Error Submit Data — GMP/BCMath & VAPID Keys

**File:** `app/Services/PushNotificationService.php` baris 24

**Error:** `ErrorException: It is highly recommended to install the GMP or BCMath extension`

**Root Cause:**
1. Ekstensi PHP `php8.4-gmp` dan `php8.4-bcmath` belum terinstall (diperlukan library `minishlink/web-push` untuk enkripsi VAPID)
2. `VAPID_PUBLIC_KEY` dan `VAPID_PRIVATE_KEY` di `.env` kosong

**Perbaikan:**
```bash
# Install ekstensi GMP dan BCMath
sudo apt install -y php8.4-gmp php8.4-bcmath

# Generate VAPID keys (otomatis update .env)
php artisan webpush:generate-keys

# Clear cache dan restart server
php artisan cache:clear
php artisan config:clear
# Restart php artisan serve
```

---

## 13. Pengembangan Fitur Rekomendasi Kumpulan Data — Sesi 14 April 2026 (Sore)

### 13.1 Latar Belakang

Setelah fitur rekomendasi organisasi berhasil, dikembangkan fitur serupa untuk field **Nama Kumpulan Data** (`group_name`). Tujuan: membantu pengguna memilih group/kategori dataset secara otomatis berdasarkan nama dataset yang diketik.

---

### 13.2 Perubahan yang Dilakukan

#### Perubahan 1 — Trigger Rekomendasi Group dari `name` (bukan `group_name`)

**File:** `resources/js/Pages/Admin/Catalog/Create.tsx`

Sebelumnya, `useEffect` rekomendasi group dipicu oleh perubahan `formData.group_name`. Diubah untuk mendengarkan perubahan `formData.name` (field Nama Data), sehingga rekomendasi muncul saat user mengetik nama dataset — bukan saat mengetik nama kumpulan data.

```tsx
// SEBELUM — dipicu oleh group_name
}, [formData.group_name]);

// SESUDAH — dipicu oleh name (Nama Data)
}, [formData.name]);
```

Dengan demikian query ke ML service juga menggunakan nilai `formData.name` bukan `formData.group_name`.

---

#### Perubahan 2 — Urutan Tampilan Form

Urutan section di form diubah agar lebih intuitif:

| Urutan | SEBELUM | SESUDAH |
|--------|---------|---------|
| 1 | Nama Data | Nama Data |
| 2 | Nama Kumpulan Data | 💡 Rekomendasi Kumpulan Data |
| 3 | 💡 Rekomendasi Kumpulan Data | Nama Kumpulan Data |
| 4 | Produsen Data | Produsen Data |

Alur yang lebih intuitif: isi Nama Data → rekomendasi muncul → klik pilih → Nama Kumpulan Data terisi otomatis.

---

#### Perubahan 3 — Card Tidak Hilang Setelah Diklik

**File:** `resources/js/Pages/Admin/Catalog/Create.tsx`

`handleSelectGroupRecommendation` sebelumnya memanggil `setGroupRecommendations([])` yang menyebabkan card menghilang setelah diklik, sehingga pengguna tidak bisa mengganti pilihan tanpa mengetik ulang.

```tsx
// SEBELUM
const handleSelectGroupRecommendation = (rec: GroupRecommendation) => {
    setFormData((prev) => ({ ...prev, group_name: rec.label }));
    setGroupRecommendations([]);  // ← menyebabkan card hilang
};

// SESUDAH
const handleSelectGroupRecommendation = (rec: GroupRecommendation) => {
    setFormData((prev) => ({ ...prev, group_name: rec.label }));
    // Tidak menghapus groupRecommendations agar card tetap tampil
};
```

---

#### Perubahan 4 — Posisi Card di Bawah Input Nama Kumpulan Data

Section `RecommendationSection` untuk group dipindahkan ke **setelah** field `group_name` (bukan sebelumnya). Sehingga tampilan akhir:

```
[ Nama Data                    ]
  ↓ (setelah 800ms debounce)
💡 Rekomendasi Kumpulan Data
  ┌──────────┐ ┌──────────┐ ┌──────────┐
  │ #1  45.7%│ │ #2   6.3%│ │ #3   6.3%│
  │frekuensi-│ │  pos     │ │teleko-   │
  │radio     │ │          │ │munikasi  │
  │ ↑ Klik   │ │ ↑ Klik   │ │ ↑ Klik   │
  └──────────┘ └──────────┘ └──────────┘
[ Nama Kumpulan Data (terisi otomatis) ]
💡 Rekomendasi Kumpulan Data (tetap tampil)
```

---

### 13.3 File yang Dimodifikasi (Sesi Sore)

| No | File | Perubahan |
|----|------|-----------|
| 1 | `app/Http/Controllers/OpenDataController.php` | Hapus filter `ckan_id` di `searchDatasetsFromDatabase()` |
| 2 | `resources/js/Pages/Admin/Catalog/Create.tsx` | Trigger group recs dari `name`; card tidak hilang; posisi card dipindah ke bawah input `group_name` |

---

*Laporan diperbarui pada 14 April 2026 pukul 15:09 WIB*
