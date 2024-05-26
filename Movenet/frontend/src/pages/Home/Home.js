import React from 'react'
import { Link } from 'react-router-dom'

import './Home.css'

export default function Home() {

    return (
        <div className='home-container'>
            <div className='home-header'>
                <h1 className='home-heading'>ZEN AI Yoga Trainer</h1>
                <Link to='/about'>
                    <button 
                        className="btn btn-secondary" 
                        id="about-btn"
                    >
                        About Us
                    </button>
                </Link>
            </div>

            <h1 className="description">Your Personal & Digital Yoga Trainer</h1>
            <div className="home-main">
                <div className="btn-section">
                    <Link to='/start'>
                        <button
                            className="btn start-btn"
                        >Get Started</button>
                    </Link>
                </div>
            </div>
        </div>
    )
}
