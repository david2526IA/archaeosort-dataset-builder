import ClassifierUploader from "@/components/ClassifierUploader";

export default function Home() {
  return (
    <div className="flex flex-1 flex-col items-center bg-zinc-50 px-6 py-16 dark:bg-black">
      <main className="flex w-full max-w-2xl flex-col items-center gap-8">
        <div className="flex flex-col items-center gap-2 text-center">
          <h1 className="text-3xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-50">
            ArchaeoSort
          </h1>
          <p className="max-w-md text-sm text-zinc-500 dark:text-zinc-400">
            Upload a photo of a stone artefact or a natural rock to classify it
            and explore visually similar images from the dataset.
          </p>
        </div>
        <ClassifierUploader />
      </main>
    </div>
  );
}
