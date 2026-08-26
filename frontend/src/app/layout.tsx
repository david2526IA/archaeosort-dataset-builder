import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ArchaeoSort",
  description:
    "Upload a photo to classify it (stone artefact vs. natural rock) and explore visually similar images, served by the ArchaeoSort inference API.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full flex flex-col font-sans">{children}</body>
    </html>
  );
}
