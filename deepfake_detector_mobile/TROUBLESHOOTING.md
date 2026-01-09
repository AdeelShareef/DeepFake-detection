# Troubleshooting Backend Connection

## Issue: App Stuck on "Uploading" or Always Getting "UNCERTAIN" Result

If the app gets stuck on "Uploading" or always returns "UNCERTAIN" immediately, it means the backend is not being reached.

### Quick Diagnosis Steps

**Step 1: Find Your Laptop's IP Address**

On your laptop (where XAMPP is running), open Command Prompt and run:
```bash
ipconfig
```

Look for your **IPv4 Address** under your active network adapter (WiFi or Ethernet). It will look like:
- `192.168.1.xxx` or
- `192.168.0.xxx` or  
- `10.0.0.xxx`

**Step 2: Test XAMPP is Running**

1. Open your browser on the laptop
2. Go to: `http://localhost/deepshield/upload_handler.php`
3. You should see an error (that's OK - it means the file exists)
4. If you get "404 Not Found", the backend files aren't in the right place

**Step 3: Test from Your Phone/Device**

On your phone's browser, try accessing:
```
http://YOUR_LAPTOP_IP/deepshield/upload_handler.php
```
Replace `YOUR_LAPTOP_IP` with the IP from Step 1.

- ✅ **If it loads:** Network is working, continue to Step 4
- ❌ **If it doesn't load:** See "Network Issues" section below

### 1. Check Backend URL

The app is currently configured to use:
```
http://192.168.100.18/deepshield
```

**For Android Emulator:**
- Change to `http://10.0.2.2/deepshield` (emulator's localhost)

**For Physical Device:**
- Use your computer's local IP address (e.g., `http://192.168.1.100/deepshield`)
- Make sure your phone and computer are on the same WiFi network

**Update in:** `lib/config/app_config.dart`

### 2. Verify Backend is Running

Make sure your Python backend is running and accessible:

```bash
# Test from command line
curl -X POST http://YOUR_IP/deepshield/upload_handler.php \
  -F "file=@test_image.jpg"
```

### 3. Check Logs

The app now has detailed logging. Run the app and check the console output:

```bash
flutter run
```

Look for:
```
=== API REQUEST ===
URL: http://...
File: ...
Size: ...

=== API RESPONSE ===
Status Code: 200
Response Data: {...}
```

### 4. Common Errors

**"Cannot connect to backend"**
- Backend is not running
- Wrong IP address
- Firewall blocking connection

**"Connection timeout"**
- Backend is too slow
- Network issues
- Wrong port

**"Invalid response format"**
- Backend returning HTML instead of JSON
- Check backend logs for errors

### 5. Network Issues - XAMPP Firewall Configuration

If your phone can't reach XAMPP on your laptop, you need to configure Windows Firewall:

**Option 1: Allow Apache Through Firewall (Recommended)**

1. Open **Windows Defender Firewall with Advanced Security**
2. Click **Inbound Rules** → **New Rule**
3. Select **Port** → Next
4. Select **TCP** and enter port **80** → Next
5. Select **Allow the connection** → Next
6. Check all profiles (Domain, Private, Public) → Next
7. Name it "Apache HTTP" → Finish

**Option 2: Temporarily Disable Firewall (Testing Only)**

⚠️ Only for testing! Re-enable after:
1. Open **Windows Security**
2. Go to **Firewall & network protection**
3. Turn off firewall for **Private network**
4. Test your app
5. **Turn it back on** when done

**Verify XAMPP Configuration:**

1. Open XAMPP Control Panel
2. Make sure **Apache** is running (green highlight)
3. Click **Config** → **Apache (httpd.conf)**
4. Find the line: `Listen 80`
5. Make sure it's NOT `Listen 127.0.0.1:80` (this would block external connections)
6. Save and restart Apache if you made changes

### 6. Test Backend Directly

Create a simple test file on your backend:

**test.php:**
```php
<?php
header('Content-Type: application/json');
echo json_encode([
    'result' => 'FAKE',
    'confidence' => 0.95,
    'fake_ratio' => 0.8,
    'frames_used' => 10,
    'media_type' => 'image'
]);
?>
```

Then temporarily change the upload endpoint in `app_config.dart`:
```dart
static const String uploadEndpoint = '/test.php';
```

If this works, the issue is with your actual upload_handler.php file.

### 7. Enable CORS (if needed)

Add to your PHP backend:
```php
header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Methods: POST');
header('Access-Control-Allow-Headers: Content-Type');
```

### 8. Check Network Permissions

Make sure AndroidManifest.xml has:
```xml
<uses-permission android:name="android.permission.INTERNET" />
```

(Already added in your app)

## Quick Fix Checklist

- [ ] Backend is running
- [ ] Correct IP address in app_config.dart
- [ ] Phone and computer on same network (for physical device)
- [ ] Backend returns valid JSON
- [ ] Check console logs for errors
- [ ] Test backend with curl/Postman first

---

# Supabase Storage Issues

## Issue: Failed to Upload Avatar - Row-Level Security Policy Error

**Error Message:**
```
Failed to upload avatar: Exception: Failed to upload avatar: 
StorageException(message: new row violates row-level security policy, 
statusCode: 403, error: Unauthorized)
```

This error occurs when trying to upload a profile avatar because the Supabase Storage bucket doesn't have the proper Row-Level Security (RLS) policies configured.

### Solution: Configure Supabase Storage Policies

#### Step 1: Access Supabase Dashboard

1. Go to [https://supabase.com/dashboard](https://supabase.com/dashboard)
2. Select your project
3. Navigate to **Storage** in the left sidebar

#### Step 2: Create or Verify the Avatars Bucket

1. Check if you have a bucket named `avatars` (or whatever your app uses)
2. If not, create one:
   - Click **New bucket**
   - Name: `avatars`
   - Public bucket: **Yes** (if you want avatars to be publicly accessible)
   - Click **Create bucket**

#### Step 3: Configure Storage Policies (Easiest Method)

**Using the Policy Wizard (Recommended):**

1. Click on the `avatars` bucket
2. Click the **Policies** tab
3. Click **New Policy** button
4. Click **"For full customization"** or **"Get started quickly"**
5. If using "Get started quickly":
   - Select **"Allow authenticated users to upload files"**
   - This will create the basic policy you need
6. Click **Review** → **Save policy**

**Manual Policy Creation (Advanced):**

If you need to create policies manually via SQL, add them **ONE AT A TIME** in the SQL Editor:

1. Go to **SQL Editor** in Supabase Dashboard
2. Create a new query
3. Paste **ONLY ONE** policy at a time
4. Run the query
5. Repeat for each policy

**Policy 1: Allow Authenticated Uploads**
```sql
CREATE POLICY "Allow authenticated uploads"
ON storage.objects
FOR INSERT
TO authenticated
WITH CHECK (bucket_id = 'avatars');
```

**Policy 2: Allow Users to Update Own Files**
```sql
CREATE POLICY "Allow users to update own files"
ON storage.objects
FOR UPDATE
TO authenticated
USING (bucket_id = 'avatars')
WITH CHECK (bucket_id = 'avatars');
```

**Policy 3: Allow Users to Delete Own Files**
```sql
CREATE POLICY "Allow users to delete own files"
ON storage.objects
FOR DELETE
TO authenticated
USING (bucket_id = 'avatars');
```

**Policy 4: Allow Public Read Access**
```sql
CREATE POLICY "Public avatar access"
ON storage.objects
FOR SELECT
TO public
USING (bucket_id = 'avatars');
```

> **⚠️ IMPORTANT:** Run each SQL statement separately! Don't paste all policies at once - this causes syntax errors.

#### Step 4: Verify Your App's Upload Code

Make sure your Flutter app is uploading files with the user ID in the path:

```dart
// Example: lib/services/storage_service.dart
final userId = supabase.auth.currentUser!.id;
final filePath = '$userId/avatar.jpg';

await supabase.storage
    .from('avatars')
    .upload(filePath, imageFile);
```

#### Alternative: Simpler Policy (Less Secure)

If you want to allow any authenticated user to upload anywhere in the bucket:

```sql
CREATE POLICY "Allow all authenticated uploads"
ON storage.objects
FOR ALL
TO authenticated
USING (bucket_id = 'avatars')
WITH CHECK (bucket_id = 'avatars');
```

⚠️ **Warning:** This is less secure as users can access/modify other users' files.

### Quick Fix via Supabase Dashboard UI

1. Go to **Storage** → **Policies**
2. Click **New Policy** → **Get started quickly**
3. Select **"Allow authenticated users to upload files"**
4. Choose the `avatars` bucket
5. Click **Review** → **Save policy**

### Verify the Fix

After configuring the policies:

1. Restart your Flutter app
2. Try uploading an avatar again
3. Check the Supabase Storage dashboard to see if the file appears

### Additional Troubleshooting

**Still getting 403 errors?**
- Verify you're logged in (check `supabase.auth.currentUser`)
- Check the file path format matches your policy
- Review Supabase logs: **Dashboard** → **Logs** → **Storage**

**Files not appearing?**
- Check bucket name matches your code
- Verify the upload path is correct
- Look for errors in Flutter console

## Issue: "Failed to pick file: Unsupported operation: _Namespace"

This error occurs on Windows when using the `file_picker` package. It's a known compatibility issue.

### Solution

The app has been updated to use `ImagePicker` instead of `FilePicker` for better Windows compatibility. 

**If you still see this error:**

1. Stop the app completely
2. Run:
   ```bash
   flutter clean
   flutter pub get
   ```
3. Restart the app:
   ```bash
   flutter run
   ```

The error should now be resolved, and you can select both images and videos without issues.
