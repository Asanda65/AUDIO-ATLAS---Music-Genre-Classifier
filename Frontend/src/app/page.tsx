'use client';
import { useState, useEffect } from 'react';
import Head from 'next/head';
import Navbar from '../components/Navbar';
import { useRouter } from 'next/router';

interface UploadedFile {
  name: string;
  size: number;
  duration: string;
}

export default function Home() {
  const [uploadedFile, setUploadedFile] = useState<UploadedFile | null>(null);
  const [genreGuessed, setGenreGuessed] = useState(false);

  useEffect(() => {
    const handlePopState = (event: PopStateEvent) => {
      if (event.state) {
        setUploadedFile(event.state.uploadedFile);
        setGenreGuessed(event.state.genreGuessed || false);
      } else {
        setUploadedFile(null);
        setGenreGuessed(false);
      }
    };

    window.addEventListener('popstate', handlePopState);

    return () => window.removeEventListener('popstate', handlePopState);
  }, []);

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      const file = event.target.files[0];
      const audio = document.createElement('audio');
      audio.src = URL.createObjectURL(file);
      audio.onloadedmetadata = () => {
        const newUploadedFile = {
          name: file.name,
          size: file.size,
          duration: formatDuration(audio.duration),
        };
        setUploadedFile(newUploadedFile);
        window.history.pushState({ uploadedFile: newUploadedFile, genreGuessed: false }, '');
      };
    }
  };

  const handleGuessGenre = () => {
    setGenreGuessed(true);
    window.history.pushState({ ...window.history.state, genreGuessed: true }, '');
  };

  const handleClick = () => {
    document.getElementById('file-upload')?.click();
  };

  const handleGuessAnother = () => {
    setUploadedFile(null);
    setGenreGuessed(false);
    window.history.pushState(null, '');
  };

  const formatDuration = (durationInSeconds: number) => {
    const minutes = Math.floor(durationInSeconds / 60);
    const seconds = Math.floor(durationInSeconds % 60);
    return `${minutes}:${seconds < 10 ? '0' : ''}${seconds}`;
  };

  return (
    <>
      <Head>
        <title>Upload Your Audio</title>
      </Head>
      <div className="flex flex-col h-screen">
        <Navbar />

        <div className="flex-grow overflow-auto">
          <div className="flex flex-col items-center justify-center h-full bg-gradient-to-r from-purple-300 via-purple-200 to-purple-300">
            {uploadedFile && !genreGuessed && (
              <div>
                <div className="flex flex-col items-center justify-center p-10 bg-white rounded-lg shadow-md md:m-0 m-4">
                  <div className="flex items-center space-x-3">
                    <span className="text-[#505083] text-2xl">🎵</span>
                    <div>
                      <p className="text-[#505083] font-semibold">{uploadedFile.name}</p>
                      <p className="text-[#505083] font-sans">Artist Name</p>
                    </div>
                  </div>
                  <p className="text-[#505083]">{uploadedFile.duration}</p>
                  <button className="mt-4 bg-[#505083] text-white py-2 px-4 rounded" onClick={handleGuessGenre}>
                    Guess The Genre
                  </button>
                </div>
              </div>
            )}
            {uploadedFile && genreGuessed && (
              <div className="space-y-6 md:m-0 m-4">
                <div className="flex flex-col items-center justify-center p-10 bg-white rounded-lg shadow-md">
                  <div className="flex items-center space-x-3">
                    <span className="text-[#505083] text-2xl">🎵</span>
                    <div>
                      <p className="text-[#505083] font-semibold">{uploadedFile.name}</p>
                      <p className="text-[#505083] font-sans">Artist Name</p>
                    </div>
                  </div>
                  <p className="text-[#505083]">{uploadedFile.duration}</p>
                </div>
                <div className="flex flex-col items-center justify-center p-10 bg-white rounded-lg shadow-md">
                  <p className="text-[#505083] mb-2">The genre of {uploadedFile.name} is...</p>

                  <div className='flex flex-col items-center justify-center'>
                    <img src='/images/hiphop-icon.png' className='pt-8 pb-8 w-40 h-40 object-contain' alt='Genre-Icon'></img>
                    <span className="text-[#505083] text-xl font-medium text-center mb-2">Hip-Hop</span> </div>

                </div>
                <div className='flex flex-col items-center justify-center'><button className="bg-[#505083] text-white py-2 px-4 rounded" onClick={handleGuessAnother}>
                  Guess another
                </button></div>
              </div>
            )}
            {!uploadedFile && (
              <div>
                <div className="flex flex-col items-center justify-center p-10 border-2 border-dotted border-[#505083] rounded-lg md:m-0 m-4">
                  <label htmlFor="file-upload" className="flex flex-col items-center cursor-pointer">
                    <span className="text-[#505083] text-4xl mb-4">🎵</span>
                    <p className="text-[#505083] mb-2 font-mono">Drag and drop an audio file (.mp3, .wav)</p>
                    <input
                      id="file-upload"
                      type="file"
                      className="hidden"
                      onChange={handleFileUpload}
                      accept=".mp3, .wav"
                    />
                  </label>
                  <div className="flex items-center mb-4">
                    <hr className="border-t-2 border-[#505083] w-12 mx-2" />
                    <span className="text-[#505083]">OR</span>
                    <hr className="border-t-2 border-[#505083] w-12 mx-2" />
                  </div>
                  <button className="bg-[#505083] text-white py-2 px-4 rounded " onClick={handleClick}>
                    Browse Files
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  );
}
