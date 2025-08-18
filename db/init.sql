-- ============= NAMESPACE =============
CREATE SCHEMA IF NOT EXISTS lab;
CREATE EXTENSION IF NOT EXISTS citext;

-- ============= AUTH =============
CREATE TABLE IF NOT EXISTS lab.roles (
  role_id       SERIAL PRIMARY KEY,
  role_name     TEXT UNIQUE NOT NULL,          -- 'reader','editor','deleter','admin'
  description   TEXT
);

CREATE TABLE IF NOT EXISTS lab.users (
  user_id       BIGSERIAL PRIMARY KEY,
  email         CITEXT UNIQUE NOT NULL,
  full_name     TEXT NOT NULL,
  is_active     BOOLEAN NOT NULL DEFAULT TRUE,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lab.user_roles (
  user_id       BIGINT REFERENCES lab.users(user_id) ON DELETE CASCADE,
  role_id       INT REFERENCES lab.roles(role_id) ON DELETE CASCADE,
  PRIMARY KEY (user_id, role_id)
);

-- ============= СПРАВОЧНИКИ =============
CREATE TABLE IF NOT EXISTS lab.standards (
  standard_id   SERIAL PRIMARY KEY,
  code          TEXT UNIQUE NOT NULL,          -- 'ASTM D790', 'ISO 6721-1', ...
  title         TEXT,
  url           TEXT,
  extra         JSONB
);

CREATE TABLE IF NOT EXISTS lab.test_types (
  test_type_id  SERIAL PRIMARY KEY,
  code          TEXT UNIQUE NOT NULL,          -- '3pB','DMTA','MASS','FTIR','MAT','HARD','COD'
  name          TEXT NOT NULL,
  method        TEXT,
  standard_id   INT REFERENCES lab.standards(standard_id),
  extra         JSONB
);

CREATE TABLE IF NOT EXISTS lab.manufacturers (
  manufacturer_id SERIAL PRIMARY KEY,
  name           TEXT NOT NULL,
  location       TEXT,
  url            TEXT,
  extra          JSONB
);

-- ============= МАТЕРИАЛЫ =============
CREATE TABLE IF NOT EXISTS lab.materials (
  material_id     BIGSERIAL PRIMARY KEY,
  material_code   TEXT UNIQUE,
  type            TEXT NOT NULL,               -- 'R-glass','Epoxy','PU','GFRP', ...
  manufacturer_id INT REFERENCES lab.manufacturers(manufacturer_id),
  density_g_cm3   NUMERIC(6,3),
  modulus_gpa     NUMERIC(6,2),
  fiber_diameter_um NUMERIC(5,2),
  sizing_type     TEXT,
  coupling_agent  TEXT,
  film_former     TEXT,
  sizing_content_wt NUMERIC(5,2),
  specific_surface_area_m2_g NUMERIC(6,3),
  total_surface_area_m2 NUMERIC(10,4),
  extra           JSONB,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lab.material_components (
  material_component_id BIGSERIAL PRIMARY KEY,
  parent_material_id BIGINT REFERENCES lab.materials(material_id) ON DELETE CASCADE,
  component_material_id BIGINT REFERENCES lab.materials(material_id),
  fraction_type   TEXT CHECK (fraction_type IN ('wt','vol','mol')),
  fraction_value  NUMERIC(6,3),
  notes           TEXT,
  extra           JSONB
);

-- ============= ПРОЕКТЫ / ОБРАЗЦЫ =============
CREATE TABLE IF NOT EXISTS lab.projects (
  project_id    BIGSERIAL PRIMARY KEY,
  code          TEXT UNIQUE NOT NULL,
  name          TEXT NOT NULL,
  owner_user_id BIGINT REFERENCES lab.users(user_id),
  description   TEXT,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  extra         JSONB
);

CREATE TABLE IF NOT EXISTS lab.samples (
  sample_id       BIGSERIAL PRIMARY KEY,
  project_id      BIGINT REFERENCES lab.projects(project_id) ON DELETE RESTRICT,
  material_id     BIGINT REFERENCES lab.materials(material_id) ON DELETE RESTRICT,
  sample_code     TEXT UNIQUE,
  geometry        JSONB,
  manufacture_date DATE,
  notes           TEXT,
  extra           JSONB,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ============= ИСПЫТАНИЯ =============
CREATE TABLE IF NOT EXISTS lab.experiments (
  experiment_id    BIGSERIAL PRIMARY KEY,
  sample_id        BIGINT REFERENCES lab.samples(sample_id) ON DELETE CASCADE,
  test_type_id     INT REFERENCES lab.test_types(test_type_id) ON DELETE RESTRICT,
  operator_user_id BIGINT REFERENCES lab.users(user_id),
  machine_id       TEXT,
  started_at       TIMESTAMPTZ,
  completed_at     TIMESTAMPTZ,
  environment      JSONB,
  raw_format       TEXT,                        -- 'CSV','PDF','HDF5','PARQUET'
  raw_data_uri     TEXT,
  raw_checksum     TEXT,
  status           TEXT CHECK (status IN ('draft','computed','final','invalid')) DEFAULT 'draft',
  notes            TEXT,
  extra            JSONB,
  created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lab.experiment_parameters (
  experiment_id   BIGINT REFERENCES lab.experiments(experiment_id) ON DELETE CASCADE,
  name            TEXT NOT NULL,
  value           TEXT,
  unit            TEXT,
  PRIMARY KEY (experiment_id, name)
);

CREATE TABLE IF NOT EXISTS lab.experiment_files (
  experiment_file_id BIGSERIAL PRIMARY KEY,
  experiment_id   BIGINT REFERENCES lab.experiments(experiment_id) ON DELETE CASCADE,
  file_role       TEXT,                        -- 'raw','report','plot','aux'
  uri             TEXT NOT NULL,
  content_type    TEXT,
  bytes_size      BIGINT,
  checksum        TEXT,
  uploaded_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
  extra           JSONB
);

-- ============= РЕЗУЛЬТАТЫ/МЕТРИКИ =============
CREATE TABLE IF NOT EXISTS lab.experiment_metrics (
  metric_id       BIGSERIAL PRIMARY KEY,
  experiment_id   BIGINT REFERENCES lab.experiments(experiment_id) ON DELETE CASCADE,
  name            TEXT NOT NULL,               -- 'youngs_modulus','max_force','Tg','ShoreD', ...
  value_num       DOUBLE PRECISION,
  value_text      TEXT,
  unit            TEXT,
  method_note     TEXT,
  extra           JSONB,
  UNIQUE (experiment_id, name)
);

-- ============= РЯДЫ ДАННЫХ =============
CREATE TABLE IF NOT EXISTS lab.experiment_timeseries (
  timeseries_id   BIGSERIAL PRIMARY KEY,
  experiment_id   BIGINT REFERENCES lab.experiments(experiment_id) ON DELETE CASCADE,
  kind            TEXT NOT NULL,               -- 'force_displacement','stress_strain','accel_freq','dmta_temp_E\''
  x_label         TEXT,
  y_label         TEXT,
  x_unit          TEXT,
  y_unit          TEXT,
  n_points        INT,
  storage_uri     TEXT,
  extra           JSONB
);

CREATE TABLE IF NOT EXISTS lab.experiment_timeseries_points (
  timeseries_id BIGINT REFERENCES lab.experiment_timeseries(timeseries_id) ON DELETE CASCADE,
  seq           INT NOT NULL,
  x_val         DOUBLE PRECISION NOT NULL,
  y_val         DOUBLE PRECISION NOT NULL,
  PRIMARY KEY (timeseries_id, seq)
);

CREATE TABLE IF NOT EXISTS lab.units (
  unit TEXT PRIMARY KEY,
  description TEXT
);

-- ============= ИНДЕКСЫ =============
CREATE INDEX IF NOT EXISTS idx_samples_project   ON lab.samples(project_id);
CREATE INDEX IF NOT EXISTS idx_samples_material  ON lab.samples(material_id);

CREATE INDEX IF NOT EXISTS idx_exp_sample        ON lab.experiments(sample_id);
CREATE INDEX IF NOT EXISTS idx_exp_testtype      ON lab.experiments(test_type_id);
CREATE INDEX IF NOT EXISTS idx_exp_started_at    ON lab.experiments(started_at);
CREATE INDEX IF NOT EXISTS idx_exp_status        ON lab.experiments(status);

CREATE INDEX IF NOT EXISTS idx_metric_exp        ON lab.experiment_metrics(experiment_id);
CREATE INDEX IF NOT EXISTS idx_metric_name       ON lab.experiment_metrics(name);

CREATE INDEX IF NOT EXISTS idx_ts_exp            ON lab.experiment_timeseries(experiment_id);
CREATE INDEX IF NOT EXISTS idx_ts_kind           ON lab.experiment_timeseries(kind);

CREATE INDEX IF NOT EXISTS idx_materials_extra   ON lab.materials USING GIN (extra);
CREATE INDEX IF NOT EXISTS idx_samples_extra     ON lab.samples USING GIN (extra);
CREATE INDEX IF NOT EXISTS idx_experiments_extra ON lab.experiments USING GIN (extra);
CREATE INDEX IF NOT EXISTS idx_metrics_extra     ON lab.experiment_metrics USING GIN (extra);

-- ============= СТАРТОВЫЕ ДАННЫЕ =============
INSERT INTO lab.standards(code,title) VALUES
  ('ASTM D790','Flexural Properties of Unreinforced and Reinforced Plastics'),
  ('ISO 6721-1','Plastics — Determination of dynamic mechanical properties'),
  ('ASTM D4065','Standard Practice for Plastics: Dynamic Mechanical Properties'),
  ('ASTM D5229','Moisture Absorption Properties'),
  ('ASTM E168','Infrared Spectroscopy'),
  ('ASTM E1876','Dynamic Young’s Modulus, Shear Modulus, and Poisson’s Ratio by Impulse Excitation'),
  ('ASTM D2240','Standard Test Method for Rubber Property—Durometer Hardness')
ON CONFLICT DO NOTHING;

INSERT INTO lab.test_types(code,name,method,standard_id)
SELECT x.code,x.name,x.method,s.standard_id
FROM (VALUES
  ('3pB','Three-Point Bending','Flexural test per ASTM D790','ASTM D790'),
  ('DMTA','Dynamic Mechanical Thermal Analysis','ISO 6721-1 / ASTM D4065','ISO 6721-1'),
  ('MASS','Mass/Water Saturation','ASTM D5229','ASTM D5229'),
  ('FTIR','Infrared Analysis','ASTM E168','ASTM E168'),
  ('MAT','Modal Analysis Test','ASTM E1876','ASTM E1876'),
  ('HARD','Hardness Shore D','ASTM D2240','ASTM D2240'),
  ('COD','Chemical Oxygen Demand',NULL,NULL)
) AS x(code,name,method,std_code)
LEFT JOIN lab.standards s ON s.code = x.std_code
ON CONFLICT (code) DO NOTHING;
