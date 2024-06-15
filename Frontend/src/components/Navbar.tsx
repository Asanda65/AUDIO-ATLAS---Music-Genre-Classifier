import Link from 'next/link';

export default function Navbar() {
    return (
        <nav className="bg-gradient-to-r from-purple-300 via-purple-200 to-purple-300 flex flex-col items-center pt-4">
            <div className="flex items-center justify-center">
                {/* bg-gradient-to-r from-purple-300 via-purple-200 to-purple-300  */}
                <img src="/images/navbar-logo.png" alt="Logo" className="h-6 w-6" />
                <Link href="/"><h1 className="text-[#191970] font-medium text-lg ml-2 font-sans">AudioAtlas</h1> </Link>
            </div>
            <hr className="border-t-[2px] border-[#191970] w-full mt-2" />
        </nav>
    );
}
