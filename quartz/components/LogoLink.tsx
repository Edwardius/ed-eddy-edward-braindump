import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "./types"

const LogoLink: QuartzComponent = ({ cfg }: QuartzComponentProps) => {
  return (
    <a href="https://ededdyedward.com" className="logo-link">
      <img src="/static/EdEddyEdward-Logo.png" alt="Eddy Zhou" />
    </a>
  )
}

LogoLink.css = `
.logo-link {
  display: inline-flex;
  background-color: #c03541;
  padding: 0.0rem 0.5rem;
  border-radius: 0.25rem;
  align-items: center;
  justify-content: center;
  transition: background-color;
  text-decoration: none;
  margin-bottom: 1rem;
}

.logo-link:hover {
  background-color: #f06571;
}

.logo-link img {
  height: 1.00rem;
  object-fit: scale-down;
  display: block;
}
`

export default (() => LogoLink) satisfies QuartzComponentConstructor