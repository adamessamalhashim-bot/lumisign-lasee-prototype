create table if not exists public.dynamic_sign_references (
  id uuid primary key default gen_random_uuid(),
  sign_name text not null,
  participant_id text not null,
  source_url text not null,
  features jsonb not null,
  metadata jsonb not null default '{}'::jsonb,
  active boolean not null default true,
  created_by text,
  created_at timestamptz not null default now()
);

create index if not exists dynamic_sign_references_sign_name_idx
  on public.dynamic_sign_references (sign_name, active);

alter table public.dynamic_sign_references enable row level security;

comment on table public.dynamic_sign_references is
  'Privacy-preserving dynamic sign templates; contains landmarks/features only, never raw videos.';
