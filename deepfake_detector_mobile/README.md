# DeepShield - Flutter Deepfake Detection Mobile App

A production-ready Flutter mobile application for detecting deepfakes in images and videos using AI-powered analysis.

## 🎯 Features

- **Image & Video Detection**: Upload and analyze images and videos for deepfake manipulation
- **Real-time Progress**: Track upload and analysis progress
- **Supabase Authentication**: Secure user authentication with email/password
- **User Profiles**: Manage profile information and avatar
- **Detection History**: View past scans with detailed results
- **Modern UI**: Dark theme with teal/cyan gradients and smooth animations
- **Responsive Design**: Works on phones and tablets in portrait and landscape modes

## 🏗️ Architecture

### Tech Stack
- **Flutter**: Latest stable version
- **State Management**: Riverpod
- **Backend**: Python (PyTorch) deepfake detection API
- **Authentication**: Supabase
- **HTTP Client**: Dio (with upload progress tracking)
- **Animations**: flutter_animate

### Project Structure
```
lib/
├── config/           # App configuration and theme
├── models/           # Data models
├── services/         # API, Auth, and Storage services
├── providers/        # Riverpod state providers
├── screens/          # UI screens
│   ├── auth/        # Login and signup
│   ├── home/        # Home screen
│   ├── detection/   # Upload and result screens
│   └── profile/     # Profile and history screens
└── widgets/          # Reusable widgets
```

## 🚀 Getting Started

### Prerequisites
- Flutter SDK (3.0.0 or higher)
- Android Studio / Xcode
- Backend API running (see Backend Setup below)

### Installation

1. **Clone the repository**
   ```bash
   cd deepfake_detector_mobile
   ```

2. **Install dependencies**
   ```bash
   flutter pub get
   ```

3. **Configure Backend URL**
   
   The app is pre-configured to use:
   - Backend API: `http://localhost/deepshield`
   - Endpoint: `/upload_handler.php`
   
   To change the backend URL, edit `lib/config/app_config.dart`:
   ```dart
   static const String baseUrl = 'YOUR_BACKEND_URL';
   ```

4. **Run the app**
   ```bash
   # For Android
   flutter run

   # For iOS (macOS only)
   flutter run -d ios
   ```

## 🔧 Backend Setup

The app connects to a Python backend that performs deepfake detection. Ensure your backend:

1. **Exposes the endpoint**: `POST /upload_handler.php`
2. **Accepts multipart file uploads**
3. **Returns JSON response**:
   ```json
   {
     "result": "FAKE | REAL | UNCERTAIN",
     "confidence": 0.0 - 1.0,
     "fake_ratio": 0.0 - 1.0,
     "frames_used": number,
     "media_type": "image | video"
   }
   ```

### File Size Limits
- **Images**: Max 5 MB (JPG, JPEG, PNG)
- **Videos**: Max 50 MB (MP4, AVI, MOV, WEBM)

## 🔐 Supabase Setup

The app uses Supabase for authentication. The credentials are pre-configured in `lib/config/app_config.dart`.

### Required Supabase Tables

Create a `profiles` table in your Supabase project:

```sql
create table profiles (
  id uuid references auth.users on delete cascade primary key,
  email text,
  name text,
  avatar_url text,
  created_at timestamp with time zone default timezone('utc'::text, now()),
  updated_at timestamp with time zone
);

-- Enable Row Level Security
alter table profiles enable row level security;

-- Create policies
create policy "Users can view their own profile"
  on profiles for select
  using (auth.uid() = id);

create policy "Users can update their own profile"
  on profiles for update
  using (auth.uid() = id);
```

## 📱 Building for Production

### Android

1. **Build APK**
   ```bash
   flutter build apk --release
   ```

2. **Build App Bundle** (for Google Play)
   ```bash
   flutter build appbundle --release
   ```

The output will be in:
- APK: `build/app/outputs/flutter-apk/app-release.apk`
- AAB: `build/app/outputs/bundle/release/app-release.aab`

### iOS

1. **Build iOS app** (requires macOS)
   ```bash
   flutter build ios --release
   ```

2. **Archive in Xcode**
   - Open `ios/Runner.xcworkspace` in Xcode
   - Select Product > Archive
   - Follow the App Store submission process

## 🎨 Design System

### Colors
- **Primary**: Teal (#14B8A6) / Cyan (#06B6D4) gradient
- **Success (Real)**: Green (#10B981)
- **Danger (Fake)**: Red (#EF4444)
- **Warning (Uncertain)**: Amber (#F59E0B)
- **Background**: Dark (#0F172A)

### Typography
- **Headings**: Bold, 24-32px
- **Body**: Regular, 16px
- **Captions**: Light, 14px

## 🧪 Testing

### Run Tests
```bash
flutter test
```

### Run with Verbose Logging
```bash
flutter run --verbose
```

## 📝 Configuration

### Change App Name
Edit in:
- `android/app/src/main/AndroidManifest.xml`: `android:label`
- `ios/Runner/Info.plist`: `CFBundleDisplayName`

### Change Package Name
```bash
flutter pub run change_app_package_name:main com.yourcompany.appname
```

## 🔍 Troubleshooting

### Common Issues

1. **"Failed to connect to backend"**
   - Ensure backend is running
   - Check `baseUrl` in `app_config.dart`
   - For Android emulator, use `10.0.2.2` instead of `localhost`

2. **"Supabase authentication failed"**
   - Verify Supabase credentials in `app_config.dart`
   - Check Supabase project is active

3. **"File picker not working"**
   - Ensure permissions are granted in device settings
   - Check AndroidManifest.xml and Info.plist have required permissions

## 📄 License

This project is part of the DeepShield deepfake detection system.

## 🤝 Support

For issues or questions, please contact the development team.
