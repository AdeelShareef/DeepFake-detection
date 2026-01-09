# Supabase Setup Instructions

## Required SQL Setup

Run these SQL commands in your Supabase SQL Editor to fix the authentication issues:

### 1. Create Profiles Table

```sql
-- Create profiles table
create table if not exists public.profiles (
  id uuid references auth.users on delete cascade primary key,
  email text,
  name text,
  avatar_url text,
  created_at timestamp with time zone default timezone('utc'::text, now()),
  updated_at timestamp with time zone
);

-- Enable Row Level Security
alter table public.profiles enable row level security;
```

### 2. Create RLS Policies (IMPORTANT!)

```sql
-- Drop existing policies if any
drop policy if exists "Users can view their own profile" on public.profiles;
drop policy if exists "Users can update their own profile" on public.profiles;
drop policy if exists "Users can insert their own profile" on public.profiles;

-- Allow users to view their own profile
create policy "Users can view their own profile"
  on public.profiles for select
  using (auth.uid() = id);

-- Allow users to insert their own profile during signup
create policy "Users can insert their own profile"
  on public.profiles for insert
  with check (auth.uid() = id);

-- Allow users to update their own profile
create policy "Users can update their own profile"
  on public.profiles for update
  using (auth.uid() = id);
```

### 3. Create Storage Bucket for Avatars (Optional)

```sql
-- Create avatars bucket
insert into storage.buckets (id, name, public)
values ('avatars', 'avatars', true);

-- Allow authenticated users to upload avatars
create policy "Users can upload their own avatar"
  on storage.objects for insert
  with check (
    bucket_id = 'avatars' AND
    auth.uid()::text = (storage.foldername(name))[1]
  );

-- Allow public access to view avatars
create policy "Avatars are publicly accessible"
  on storage.objects for select
  using (bucket_id = 'avatars');
```

## Verification

After running the SQL commands:

1. Go to Supabase Dashboard → Authentication → Policies
2. Verify that the `profiles` table has 3 policies:
   - ✅ Users can view their own profile
   - ✅ Users can insert their own profile
   - ✅ Users can update their own profile

3. Test signup in the app - it should now work!

## Common Issues

### "new row violates row-level security policy"
- **Cause**: Missing INSERT policy on profiles table
- **Fix**: Run the RLS policies SQL above

### "Failed to fetch user profile"
- **Cause**: Profile doesn't exist in database
- **Fix**: The app now handles this automatically by creating the profile

### "Account already exists"
- **Cause**: User trying to sign up with existing email
- **Fix**: App now shows clear message to log in instead
