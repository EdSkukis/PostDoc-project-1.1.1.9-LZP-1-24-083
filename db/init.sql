CREATE SCHEMA IF NOT EXISTS lab;
CREATE EXTENSION IF NOT EXISTS citext;

CREATE TABLE IF NOT EXISTS lab.projects(
  project_id BIGSERIAL PRIMARY KEY,
  code TEXT UNIQUE NOT NULL,
  name TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lab.test_types(
  test_type_id SERIAL PRIMARY KEY,
  code TEXT UNIQUE NOT NULL,
  name TEXT NOT NULL
);

INSERT INTO lab.test_types(code,name) VALUES
  ('3pB','Three-Point Bending'),
  ('DMTA','Dynamic Mechanical Thermal Analysis'),
  ('MASS','Mass/Water Saturation'),
  ('FTIR','Infrared Analysis'),
  ('MAT','Modal Analysis Test'),
  ('HARD','Hardness Shore D')
ON CONFLICT DO NOTHING;

CREATE TABLE IF NOT EXISTS lab.materials(
  material_id BIGSERIAL PRIMARY KEY,
  material_code TEXT UNIQUE,
  type TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lab.material_components(
  material_component_id BIGSERIAL PRIMARY KEY,
  parent_material_id BIGINT REFERENCES lab.materials(material_id) ON DELETE CASCADE,
  component_material_id BIGINT REFERENCES lab.materials(material_id),
  fraction_type TEXT CHECK (fraction_type IN ('wt','vol','mol')),
  fraction_value NUMERIC(6,3),
  notes TEXT
);

CREATE TABLE IF NOT EXISTS lab.samples(
  sample_id BIGSERIAL PRIMARY KEY,
  project_id BIGINT REFERENCES lab.projects(project_id) ON DELETE RESTRICT,
  material_ref_id BIGINT NOT NULL,
  material_ref_type TEXT NOT NULL CHECK (material_ref_type IN ('material','material_component')),
  sample_code TEXT UNIQUE,
  geometry JSONB,
  manufacture_date DATE,
  notes TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lab.experiments(
  experiment_id BIGSERIAL PRIMARY KEY,
  sample_id BIGINT REFERENCES lab.samples(sample_id) ON DELETE CASCADE,
  test_type_id INT REFERENCES lab.test_types(test_type_id) ON DELETE RESTRICT,
  started_at TIMESTAMPTZ,
  completed_at TIMESTAMPTZ,
  status TEXT DEFAULT 'draft'
);

CREATE INDEX IF NOT EXISTS idx_samples_project ON lab.samples(project_id);
CREATE INDEX IF NOT EXISTS idx_experiments_sample ON lab.experiments(sample_id);
CREATE INDEX IF NOT EXISTS idx_experiments_testtype ON lab.experiments(test_type_id);
